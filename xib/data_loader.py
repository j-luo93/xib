import ast
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from arglib import add_argument, init_g_attr
from devlib import PandasDataset, get_range

LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]


@dataclass
class Batch:
    feat_matrix: LongTensor
    pos_to_predict: LongTensor
    target_feat: LongTensor = field(init=False)

    def __post_init__(self):
        batch_i = get_range(self.batch_size, 2, 0)
        window_i = get_range(self.window_size, 2, 1)
        self.target_feat = feat_matrix[batch_i, window_i, self.pos_to_predict]

    @property
    def shape(self):
        return self.feat_matrix.shape

    @property
    def batch_size(self):
        return self.feat_matrix.size(0)

    @property
    def window_size(self):
        return self.feat_matrix.size(1)


@init_g_attr
class IpaDataLoader(DataLoader):

    add_argument('batch_size', default=16, dtype=int, msg='batch size')
    add_argument('num_workers', default=5, dtype=int, msg='number of workers for the data loader')
    add_argument('data_path', dtype=str, msg='path to the feat data in tsv format.')

    def __init__(self, data_path: 'a', batch_size, num_workers):
        df = pd.read_csv(data_path, sep='\t', converts={'feat_matrix': ast.literal_eval})
        super().__init__(df, columns=['feat_matrix'], batch_size=batch_size,
                         shuffle=True, num_workers=num_workers)

    def __iter__(self):
        for batch in super().__iter__():
            bs, ws, _ = batch.feat_matrix.shape
            feat_matrix = batch.feat_matrix.repeat(ws, 1, 1)
            pos_to_predict = get_range(ws, 2, 1).repeat(bs, 1)
            return Batch(feat_matrix, pos_to_predict)
