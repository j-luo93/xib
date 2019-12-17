import logging
import pickle
import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from dev_misc import LT, add_condition, g
from dev_misc.arglib import add_argument, g, init_g_attr
from dev_misc.devlib import (batch_class, get_array, get_length_mask,
                             get_range, get_zeros)
from dev_misc.trainlib import has_gpus
from dev_misc.trainlib.base_data_loader import (BaseDataLoader,
                                                BaseDataLoaderRegistry)
from dev_misc.trainlib.tracker.tracker import Task
from dev_misc.utils import cached_property
from xib.batch import CbowIpaBatch
from xib.ipa import Category, Index, conditions, get_enum_by_cat
from xib.ipa.process import BaseSegment, Segment, SegmentWindow
from xib.training.task import Task

from .batch import BaseBatch, DenseIpaBatch, IpaBatch


class BaseDataset(Dataset, metaclass=ABCMeta):

    cache_suffix = 'cache'

    def __init__(self, data_path: Path):
        cache_path = Path(str(data_path) + f'.{self.cache_suffix}')
        if cache_path.exists():
            # with cache_path.open('rb') as fin:
            #     self.data = pickle.load(fin)
            self.data = torch.load(cache_path)
            path = cache_path
        else:
            self.data = self.load_data(data_path)
            torch.save(self.data, cache_path)
            # with cache_path.open('wb') as fout:
            #     pickle.dump(self.data, fout, protocol=2)
            path = data_path
        logging.info(f'Loaded {len(self)} segments in total from {path}.')

    @abstractmethod
    def load_data(self, data_path: Path) -> Dict: ...

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


class IpaDataset(BaseDataset):

    # add_argument('use_cached_pth', default=False, dtype=bool,
    #              msg='Flag to use precomputed pth file instead of processing the data on the file.')

    def load_data(self, data_path: Path):
        segments = self._get_segment_dict(data_path)
        return {
            'segments': get_array(list(segments.keys())),
            'matrices': [segment.feat_matrix for segment in segments.values()]
        }


@dataclass
class CollateReturn:
    segments: np.ndarray
    lengths: torch.LongTensor
    matrices: torch.LongTensor
    gold_tag_seqs: Optional[torch.LongTensor] = None
    unit_id_seqs: Optional[torch.LongTensor] = None


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
    unit_id_seqs = None
    if 'unit_id_seq' in batch[0]:
        unit_id_seqs = collate_helper('unit_id_seq', torch.Tensor, pad=True)
    return CollateReturn(segments, lengths, matrices, gold_tag_seqs=gold_tag_seqs, unit_id_seqs=unit_id_seqs)


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


add_argument('max_segment_length', default=10, dtype=int,
             msg='Max length for segments. Longer ones will be broken down into moving windows.')
add_argument('broken_words', default=False, dtype=bool, msg='Flag to break words down.')


class UnbrokenIpaDataset(IpaDataset):

    cache_suffix = 'unbroken.cache'

    def load_data(self, data_path: Path):
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
                start = 0
                while start < len(tokens):
                    # for start in range(len(tokens)):
                    while end < len(tokens) and cum_lengths[end] - ex_cum_lengths[start] <= g.max_segment_length:
                        end += 1
                    if end <= start:
                        end = start + 1
                        start += 1
                        continue
                    if end > last_end:
                        segment_window = segments[start: end]
                        if len(SegmentWindow(segment_window)) >= g.min_word_length:
                            segment_windows.append(segment_window)
                    last_end = end
                    start = last_end
        return {
            'segments': get_array(segment_windows),
            'matrices': [torch.cat([segment.feat_matrix for segment in segment_window], dim=0) for segment_window in segment_windows]
        }

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        ret['segment'] = SegmentWindow(ret['segment'])
        ret['gold_tag_seq'] = ret['segment'].gold_tag_seq
        return ret


class BrokenIpaDataset(IpaDataset):

    @property
    def window_size(self):
        return g.max_segment_length

    def load_data(self, data_path: Path):
        segment_dict = self._get_segment_dict(data_path)
        segment_windows = list()
        with data_path.open('r', encoding='utf8') as fin:
            for line in fin:
                tokens = line.strip().split()
                segments = [segment_dict[token] for token in tokens]
                sw = SegmentWindow(segments)
                start = 0
                while True:
                    end = min(start + self.window_size, len(sw))
                    broken_sw = sw.break_segment(start, end - 1)
                    segment_windows.append(broken_sw)
                    if end >= len(sw):
                        break
                    start += 1

        return {
            'segments': get_array(segment_windows),
            'matrices': [sw.feat_matrix for sw in segment_windows]
        }

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        ret['gold_tag_seq'] = ret['segment'].gold_tag_seq
        return ret


class CbowIpaDataset(BrokenIpaDataset):

    cache_suffix = 'cbow.cache'

    @property
    def window_size(self):
        return g.window_size


add_argument('input_format', default='ipa', dtype=str, choices=['ipa', 'text'], msg='Input format to use.')
add_condition('input_format', 'text', 'dense_input', True)


class UnbrokenTextDataset(UnbrokenIpaDataset):
    """This only produces batches of texts, not ipas."""

    cache_suffix = 'unbroken.text.cache'

    def load_data(self, data_path: Path):
        data = super().load_data(data_path)

        self._set_unit_ids(data)

        matrices = list()
        for segment in data['segments']:
            sw = SegmentWindow(segment)
            unit_inds = torch.LongTensor([self.unit2id[cv] for cv in sw.cv_list])
            matrices.append(unit_inds)
        data['unit_id_seqs'] = matrices
        return data

    def _set_unit_ids(self, data=None):
        # HACK(j_luo) Quite hacky.
        data = data or self.data
        if not hasattr(self, 'id2unit'):
            units = set()
            for segment in data['segments']:
                sw = SegmentWindow(segment)
                units.update(sw.cv_list)
            self.id2unit = sorted(units)
            self.unit2id = {u: i for i, u in enumerate(self.id2unit)}

    @cached_property
    def unit_vocab_size(self):
        self._set_unit_ids()
        return len(self.id2unit)

    def __getitem__(self, idx):
        ret = super().__getitem__(idx)
        ret['unit_id_seq'] = self.data['unit_id_seqs'][idx]
        return ret


total = Index.total_indices()
_g2f = torch.LongTensor(total)
indices = [Index.get_feature(i).value for i in range(total)]
for index in indices:
    _g2f[index.g_idx] = index.f_idx


DenseFeatureMatrix = Dict[Category, torch.FloatTensor]


def convert_to_dense(feat_matrix: LT) -> DenseFeatureMatrix:
    names = feat_matrix.names
    bs = feat_matrix.size('batch')
    ml = feat_matrix.size('length')
    fm = _g2f[feat_matrix.rename(None)].refine_names(*names)
    dfms = dict()
    for cat in Category:
        e = get_enum_by_cat(cat)
        dfm_idx = fm[..., cat.value]
        dfm = get_zeros(bs, ml, len(e), cpu=True)
        dfm = dfm.scatter(2, dfm_idx.rename(None).unsqueeze(dim=-1), 1.0)
        dfms[cat] = dfm.refine_names('batch', 'length', f'{cat.name}_feat')
    if has_gpus():
        dfms = {k: v.cuda() for k, v in dfms.items()}
    return dfms


@batch_class
class ContinuousIpaBatch(BaseBatch):
    gold_tag_seqs: Optional[torch.LongTensor] = None
    dense_feat_matrix: Optional[DenseFeatureMatrix] = field(init=False, default=None)

    def _post_init_helper(self):
        super()._post_init_helper()
        self.gold_tag_seqs.rename_('batch', 'length')

        if g.dense_input:
            self.dense_feat_matrix = convert_to_dense(self.feat_matrix)


@batch_class
class UnbrokenTextBatch(ContinuousIpaBatch, BaseBatch):
    # TODO(j_luo) This is actually not optional. Follow https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses to rewrite it.
    unit_id_seqs: Optional[LT] = None

    def _post_init_helper(self):
        super()._post_init_helper()
        self.unit_id_seqs.rename_('batch', 'length')


class UnbrokenIpaDataLoader(IpaDataLoader):

    batch_cls = ContinuousIpaBatch
    dataset_cls = UnbrokenIpaDataset

    def _prepare_batch(self, collate_return: CollateReturn) -> ContinuousIpaBatch:
        cls = type(self)
        batch_cls = cls.batch_cls
        return batch_cls(
            collate_return.segments,
            collate_return.lengths,
            collate_return.matrices,
            gold_tag_seqs=collate_return.gold_tag_seqs,
        )


class UnbrokenTextDataLoader(UnbrokenIpaDataLoader):

    batch_cls = UnbrokenTextBatch
    dataset_cls = UnbrokenTextDataset

    def _prepare_batch(self, collate_return: CollateReturn) -> UnbrokenTextBatch:
        cls = type(self)
        batch_cls = cls.batch_cls
        return batch_cls(
            collate_return.segments,
            collate_return.lengths,
            collate_return.matrices,
            gold_tag_seqs=collate_return.gold_tag_seqs,
            unit_id_seqs=collate_return.unit_id_seqs,
        )

    def __iter__(self):
        for batch in super().__iter__():
            if batch.max_length < g.min_word_length:
                continue
            yield batch.cuda()


class BrokenIpaDataLoader(UnbrokenIpaDataLoader):

    dataset_cls = BrokenIpaDataset


class CbowIpaDataLoader(BrokenIpaDataLoader):

    dataset_cls = CbowIpaDataset
    batch_cls = CbowIpaBatch

    def _prepare_batch(self, collate_return: CollateReturn) -> CbowIpaBatch:
        cls = type(self)
        batch_cls = cls.batch_cls
        return batch_cls(
            collate_return.segments,
            collate_return.lengths,
            collate_return.matrices,
        )


@batch_class
class DenseCbowIpaBatch(CbowIpaBatch):
    dense_feat_matrix: DenseFeatureMatrix = field(init=False)

    def _post_init_helper(self):
        super()._post_init_helper()
        self.dense_feat_matrix = convert_to_dense(self.feat_matrix)


class DenseCbowIpaDataLoader(CbowIpaDataLoader):
    batch_cls = DenseCbowIpaBatch


ContinuousTextDataLoader = Union[BrokenIpaDataLoader, UnbrokenIpaDataLoader]


class DataLoaderRegistry(BaseDataLoaderRegistry):

    def get_data_loader(self, task: Task, data_path: Path):
        if task.name in ['lm', 'mlm']:
            dl = IpaDataLoader(data_path, task)
        elif task.name == 'cbow':
            dl = CbowIpaDataLoader(data_path, task)
        elif task.name == 'adapt_cbow':
            dl = DenseCbowIpaDataLoader(data_path, task)
        elif task.name == 'adapt_lm':
            dl = DenseIpaDataLoader(data_path, task)
        elif task.name in ['decipher', 'transfer', 'extract']:
            if g.input_format == 'text':
                dl_cls = UnbrokenTextDataLoader
            else:
                dl_cls = BrokenIpaDataLoader if g.broken_words else UnbrokenIpaDataLoader
            dl = dl_cls(data_path, task)
        else:
            raise ValueError(f'Unsupported task {task.name}.')
        return dl
