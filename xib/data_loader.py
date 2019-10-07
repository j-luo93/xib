import ast
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Union

import numpy as np
import pandas as pd
import torch

from arglib import add_argument, init_g_attr
from devlib import (PandasDataLoader, get_length_mask, get_range, get_tensor,
                    pad_to_dense, pandas_collate_fn)

LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]


@dataclass
class Batch:
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


FEAT_COLS = ['f1', 'f2', 'f3']


@init_g_attr(default='none')  # NOTE(j_luo) many attributes are handled as properties later by DataLoader.
class IpaDataLoader(PandasDataLoader):

    add_argument('batch_size', default=16, dtype=int, msg='batch size')
    add_argument('num_workers', default=5, dtype=int, msg='number of workers for the data loader')
    add_argument('data_path', dtype=str, msg='path to the feat data in tsv format.')
    add_argument('window_size', default=3, dtype=int, msg='window size for the cnn kernel')

    def __init__(self, data_path: 'p', batch_size, num_workers):
        converters = {col: ast.literal_eval for col in FEAT_COLS}
        df = pd.read_csv(data_path, sep='\t', converters=converters)
        super().__init__(df, columns=['segment'] + FEAT_COLS, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers)

    def __iter__(self):
        for batch_df in super().__iter__():
            segment = batch_df['segment'].values
            feats = [get_tensor(pad_to_dense(batch_df[col], dtype='int64')) for col in FEAT_COLS]
            feat_matrix = torch.stack(feats, dim=-1)
            bs, ws, _ = feat_matrix.shape
            target_weight = get_length_mask(list(map(len, segment)), feat_matrix.size(1))

            feat_matrix = feat_matrix.repeat(ws, 1, 1)
            pos_to_predict = get_range(ws, 2, 0).repeat(1, bs).view(-1)
            target_weight = target_weight.repeat(ws, 1, 1)
            yield Batch(feat_matrix, target_weight, pos_to_predict)
