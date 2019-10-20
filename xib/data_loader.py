import logging
import random
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from arglib import add_argument, g, init_g_attr
from devlib import get_length_mask, get_range, get_tensor
from xib.ipa import Category, Index, Ptype, conditions

LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]


@dataclass
class Batch:
    segments: np.ndarray
    feat_matrix: LongTensor
    target_weight: LongTensor
    pos_to_predict: LongTensor
    target_feat: LongTensor = field(init=False)

    _g2f = None

    def __post_init__(self):
        batch_i = get_range(self.batch_size, 1, 0)
        # NOTE(j_luo) This is global index.
        target_feat = self.feat_matrix[batch_i, self.pos_to_predict]
        # Get conversion matrix.
        if self._g2f is None:
            total = Index.total_indices()
            self._g2f = torch.LongTensor(total)
            indices = [Index.get_feature(i).value for i in range(total)]
            for index in indices:
                self._g2f[index.g_idx] = index.f_idx
        # NOTE(j_luo) This is feature index.
        self.target_feat = self._g2f[target_feat]
        # NOTE(j_luo) This has to be a new copy, therefore expand won't work.
        self.target_weight = self.target_weight.unsqueeze(dim=-1).repeat(1, g.num_feature_groups)

        # NOTE(j_luo) If the condition is not satisfied, the target weight should be set to 0.
        for cat, index in conditions.items():
            idx = cat.value
            condition_idx = index.f_idx
            mask = condition_idx != self.target_feat[:, index.c_idx]
            self.target_weight[mask, idx] = 0.0

        # NOTE(j_luo) Refine names.
        # IDEA(j_luo) We can move this process a bit earlier to DataLoader?
        self.feat_matrix = self.feat_matrix.refine_names('batch', 'length', 'feat_group')
        self.target_weight = self.target_weight.refine_names('batch', 'feat_group')
        self.pos_to_predict = self.pos_to_predict.refine_names('batch')
        self.target_feat = self.target_feat.refine_names('batch', 'feat_group')

        for attr, anno in self.__annotations__.items():
            if anno is not np.ndarray:
                setattr(self, attr, get_tensor(getattr(self, attr)))

    def __len__(self):
        return self.batch_size

    @property
    def shape(self):
        return self.feat_matrix.shape

    @property
    def batch_size(self):
        return self.feat_matrix.size(0)

    @property
    def window_size(self):
        return self.feat_matrix.size(1)


class IpaDataset(Dataset):

    def __init__(self, data_path):
        self.data = torch.load(data_path)
        logging.info(f'Loaded {len(self)} segments in total.')

    def __len__(self):
        return len(self.data['segments'])

    def __getitem__(self, idx):
        return self.data['segments'][idx], self.data['matrices'][idx]


def collate_fn(batch):
    segments, matrices = zip(*batch)
    lengths = torch.LongTensor(list(map(len, matrices)))
    feat_matrix = torch.nn.utils.rnn.pad_sequence(matrices, batch_first=True)  # size: sl x K -> bs x max_sl x K
    return np.asarray(segments), feat_matrix, lengths


@init_g_attr
class BatchSampler(Sampler):

    def __init__(self, dataset: 'a', char_per_batch: 'p', shuffle: 'p' = True):
        self.dataset = dataset
        # Partition the entire dataset beforehand into batches by length.
        lengths = np.asarray(list(map(len, self.dataset.data['segments'])))
        indices = lengths.argsort()
        logging.info('Partitioning the data into batches.')
        self.idx_batches = list()
        i = 0
        while i < len(indices):
            max_len = lengths[indices[i]]
            bs = char_per_batch // max_len
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


add_argument('data_path', dtype=str, msg='path to the feat data in tsv format.')
add_argument('num_workers', default=5, dtype=int, msg='number of workers for the data loader')
add_argument('char_per_batch', default=500, dtype=int, msg='batch_size')


@init_g_attr
class IpaDataLoader(DataLoader):

    add_argument('window_size', default=3, dtype=int, msg='window size for the cnn kernel')

    def __init__(self, data_path: 'p', char_per_batch: 'p', num_workers):
        dataset = IpaDataset(data_path)
        batch_sampler = BatchSampler(dataset, char_per_batch, shuffle=True)
        super().__init__(dataset, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn)

    def __iter__(self):
        for segments, feat_matrix, lengths in super().__iter__():
            new_lengths = np.asarray(list(map(lambda x: len(x.split('-')), segments)))
            bs, ws, _ = feat_matrix.shape
            target_weight = get_length_mask(lengths, ws)

            feat_matrix = feat_matrix.repeat(ws, 1, 1)
            pos_to_predict = get_range(ws, 2, 0).repeat(1, bs).view(-1)
            target_weight = target_weight.t().reshape(-1)
            batch = Batch(segments, feat_matrix, target_weight, pos_to_predict)
            yield batch


@init_g_attr
class ContinuousTextDataLoader(IpaDataLoader):

    # TODO(j_luo) might want to add curriculum learning to anneal the window size.
    pass
