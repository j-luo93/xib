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
    pos_to_predict: LongTensor
    target_feat: LongTensor = field(init=False)
    target_weight: LongTensor = field(init=False)

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
        self.target_weight = torch.ones(self.batch_size, g.num_feature_groups)

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
    segments = list()
    matrices = list()
    positions = list()
    lengths = list()
    for segment, matrix in batch:
        length = len(matrix)
        for position in range(length):
            segments.append(segment)
            matrices.append(matrix)
            positions.append(position)
            lengths.append(length)
    matrices = torch.nn.utils.rnn.pad_sequence(matrices, batch_first=True)
    return np.asarray(segments), matrices, torch.LongTensor(positions), torch.LongTensor(lengths)


@init_g_attr
class BatchSampler(Sampler):
    """
    This class works by sampling (randomly if shuffled) segments, until the total number of characters exceeds char_per_batch.
    Note that __len__ is not defined.
    """

    def __init__(self, dataset: 'a', char_per_batch: 'p', shuffle: 'p' = True):
        self.dataset = dataset
        self.lengths = np.asarray(list(map(len, self.dataset.data['matrices'])))

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        total_num_char = 0
        batch = list()
        for idx in indices:
            length = self.lengths[idx]
            if length + total_num_char > self.char_per_batch:
                yield batch
                batch = [idx]
                total_num_char = length
            else:
                batch.append(idx)
                total_num_char += length
        if batch:
            yield batch


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
        for segments, feat_matrix, pos_to_predict, lengths in super().__iter__():
            bs, ws, _ = feat_matrix.shape
            batch = Batch(segments, feat_matrix, pos_to_predict)
            yield batch


@init_g_attr
class ContinuousTextDataLoader(IpaDataLoader):

    # TODO(j_luo) might want to add curriculum learning to anneal the window size.
    pass
