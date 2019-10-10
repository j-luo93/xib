import ast
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from arglib import add_argument, init_g_attr
from devlib import (PandasDataLoader, get_length_mask, get_range, get_tensor,
                    pad_to_dense, pandas_collate_fn)

LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]


@dataclass
class Batch:
    segments: np.ndarray
    feat_matrix: LongTensor
    target_weight: LongTensor
    pos_to_predict: LongTensor
    target_feat: LongTensor = field(init=False)

    def __post_init__(self):
        batch_i = get_range(self.batch_size, 1, 0)
        self.target_feat = self.feat_matrix[batch_i, self.pos_to_predict]

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

    def __len__(self):
        return len(self.data['segments'])

    def __getitem__(self, idx):
        return self.data['segments'][idx], self.data['matrices'][idx]


def collate_fn(batch):
    segments, matrices = zip(*batch)
    lengths = torch.LongTensor(list(map(len, matrices)))
    feat_matrix = torch.nn.utils.rnn.pad_sequence(matrices, batch_first=True)  # size: sl x K -> bs x max_sl x K
    return np.asarray(segments), feat_matrix, lengths


@init_g_attr(default='none')  # NOTE(j_luo) many attributes are handled as properties later by DataLoader.
class IpaDataLoader(DataLoader):

    add_argument('batch_size', default=16, dtype=int, msg='batch size')
    add_argument('num_workers', default=5, dtype=int, msg='number of workers for the data loader')
    add_argument('data_path', dtype=str, msg='path to the feat data in tsv format.')
    add_argument('window_size', default=3, dtype=int, msg='window size for the cnn kernel')

    def __init__(self, data_path: 'p', batch_size, num_workers):
        dataset = IpaDataset(data_path)
        super().__init__(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    def __iter__(self):
        for segments, feat_matrix, lengths in super().__iter__():
            bs, ws, _ = feat_matrix.shape
            target_weight = get_length_mask(lengths, ws)

            feat_matrix = feat_matrix.repeat(ws, 1, 1)
            pos_to_predict = get_range(ws, 2, 0).repeat(1, bs).view(-1)
            target_weight = target_weight.t().reshape(-1)
            yield Batch(segments, feat_matrix, target_weight, pos_to_predict)
