import logging
import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from dev_misc import g
from dev_misc.arglib import add_argument, g, init_g_attr
from dev_misc.devlib import (batch_class, get_array, get_length_mask,
                             get_range, get_zeros)
from dev_misc.trainlib.base_data_loader import (BaseDataLoader,
                                                BaseDataLoaderRegistry)
from dev_misc.trainlib.tracker.tracker import Task
from xib.ipa import Category, Index, conditions, get_enum_by_cat
from xib.ipa.process import Segment, SegmentWindow
from xib.training.task import Task

from .batch import BaseBatch, DenseIpaBatch, IpaBatch


class IpaDataset(Dataset):

    add_argument('use_cached_pth', default=False, dtype=bool,
                 msg='Flag to use precomputed pth file instead of processing the data on the file.')

    def __init__(self, data_path: Path):
        if g.use_cached_pth:
            self.data = torch.load(data_path)
        else:
            segments = self._get_segment_dict(data_path)
            self.data = {
                'segments': get_array(list(segments.keys())),
                'matrices': [segment.feat_matrix for segment in segments.values()]
            }
        logging.info(f'Loaded {len(self)} segments in total.')

    def _get_segment_dict(self, data_path: Path) -> Dict[str, Segment]:
        segments = dict()
        with data_path.open('r', encoding='utf8') as fin:
            for line in fin:
                tokens = line.strip().split()
                for token in tokens:
                    if token not in segments:
                        segments[token] = Segment(token)
        return segments

    def __len__(self):
        return len(self.data['segments'])

    def __getitem__(self, idx):
        segment = self.data['segments'][idx]
        matrix = self.data['matrices'][idx]
        length = len(matrix)
        return {
            'segment': segment,
            'matrix': matrix,
            'length': length
        }


@dataclass
class CollateReturn:
    segments: np.ndarray
    lengths: torch.LongTensor
    matrices: torch.LongTensor
    gold_tag_seqs: Optional[torch.LongTensor] = None


def collate_fn(batch) -> CollateReturn:

    def collate_helper(key, cls, pad=False):
        ret = [item[key] for item in batch]
        if cls is np.ndarray:
            return get_array(ret)
        elif cls is torch.Tensor:
            if pad:
                ret = torch.nn.utils.rnn.pad_sequence(ret, batch_first=True)
            else:
                ret = torch.LongTensor(ret)
            return ret
        else:
            raise ValueError(f'Unsupported class "{cls}".')

    segments = collate_helper('segment', np.ndarray)
    matrices = collate_helper('matrix', torch.Tensor, pad=True)
    lengths = collate_helper('length', torch.Tensor)
    gold_tag_seqs = None
    if 'gold_tag_seq' in batch[0]:
        gold_tag_seqs = collate_helper('gold_tag_seq', torch.Tensor, pad=True)
    return CollateReturn(segments, lengths, matrices, gold_tag_seqs=gold_tag_seqs)


class BatchSampler(Sampler):

    def __init__(self, dataset: Dataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        # Partition the entire dataset beforehand into batches by length.
        lengths = np.asarray(list(map(len, self.dataset.data['matrices'])))
        indices = lengths.argsort()[::-1]  # NOTE(j_luo) Sort in descending order.
        logging.info('Partitioning the data into batches.')
        self.idx_batches = list()
        i = 0
        while i < len(indices):
            max_len = lengths[indices[i]]
            bs = g.char_per_batch // max_len
            if bs == 0:
                raise RuntimeError(f'Batch too small!')
            self.idx_batches.append(indices[i: i + bs])
            i += bs

    def __len__(self):
        return len(self.idx_batches)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.idx_batches)
        yield from self.idx_batches


class BaseIpaDataLoader(BaseDataLoader, metaclass=ABCMeta):

    add_argument('data_path', dtype='path', msg='path to the feat data in tsv format.')
    add_argument('num_workers', default=0, dtype=int, msg='number of workers for the data loader')
    add_argument('char_per_batch', default=500, dtype=int, msg='batch_size')
    add_argument('new_style', default=False, dtype=bool, msg='flag to use new style ipa annotations')

    dataset_cls: Type[Dataset]

    def __init__(self, data_path: Path, task: Task):
        dataset = type(self).dataset_cls(data_path)
        batch_sampler = BatchSampler(dataset, shuffle=True)
        cls = type(self)
        super().__init__(dataset, task, batch_sampler=batch_sampler, pin_memory=True,
                         num_workers=g.num_workers, collate_fn=collate_fn)

    @abstractmethod
    def _prepare_batch(self, collate_return: CollateReturn) -> BaseBatch:
        pass

    def __iter__(self):
        for collate_return in super().__iter__():
            batch = self._prepare_batch(collate_return)
            yield batch.cuda()


class IpaDataLoader(BaseIpaDataLoader):

    batch_cls: Type[BaseBatch] = IpaBatch
    dataset_cls: Type[Dataset] = IpaDataset

    def _prepare_batch(self, collate_return: CollateReturn) -> IpaBatch:
        cls = type(self)
        batch_cls = cls.batch_cls
        return batch_cls(collate_return.segments, collate_return.lengths, collate_return.matrices)


class DenseIpaDataLoader(IpaDataLoader):
    batch_cls = DenseIpaBatch


class ContinuousTextIpaDataset(IpaDataset):

    add_argument('max_segment_length', default=10, dtype=int,
                 msg='Max length for segments. Longer ones will be broken down into moving windows.')

    def __init__(self, data_path: Path):
        segment_dict = self._get_segment_dict(data_path)
        segment_windows = list()
        with data_path.open('r', encoding='utf8') as fin:
            for line in fin:
                tokens = line.strip().split()
                segments = [segment_dict[token] for token in tokens]
                lengths = np.asarray([len(segment) for segment in segments])
                cum_lengths = np.cumsum(lengths)
                ex_cum_lengths = np.concatenate([np.zeros([1], dtype=np.int32), cum_lengths[:-1]])
                last_end = -1
                end = 0
                for start in range(len(tokens)):
                    while end < len(tokens) and cum_lengths[end] - ex_cum_lengths[start] <= g.max_segment_length:
                        end += 1
                    if end <= start:
                        end = start + 1
                        continue
                    if end > last_end:
                        segment_window = segments[start: end]
                        segment_windows.append(segment_window)
                    last_end = end
        self.data = {
            'segments': get_array(segment_windows),
            'matrices': [torch.cat([segment.feat_matrix for segment in segment_window], dim=0) for segment_window in segment_windows]
        }
        logging.info(f'Loaded {len(self)} segments in total.')

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        ret['segment'] = SegmentWindow(ret['segment'])
        ret['gold_tag_seq'] = ret['segment'].gold_tag_seq
        return ret


@batch_class
class ContinuousTextIpaBatch(BaseBatch):
    gold_tag_seqs: Optional[torch.LongTensor] = None

    def _post_init_helper(self):
        super()._post_init_helper()
        self.gold_tag_seqs.rename_('batch', 'length')


class ContinuousTextDataLoader(IpaDataLoader):

    batch_cls = ContinuousTextIpaBatch
    dataset_cls = ContinuousTextIpaDataset

    def _prepare_batch(self, collate_return: CollateReturn) -> ContinuousTextIpaBatch:
        cls = type(self)
        batch_cls = cls.batch_cls
        return batch_cls(
            collate_return.segments,
            collate_return.lengths,
            collate_return.matrices,
            gold_tag_seqs=collate_return.gold_tag_seqs,
        )


class DataLoaderRegistry(BaseDataLoaderRegistry):

    def get_data_loader(self, task: Task, data_path: Path):
        if task.name in ['lm', 'mlm']:
            dl = IpaDataLoader(data_path, task)
        elif task.name in ['decipher', 'transfer']:
            dl = ContinuousTextDataLoader(data_path, task)
        else:
            raise ValueError(f'Unsupported task {task.name}.')
        return dl
