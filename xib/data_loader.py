from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from arglib import add_argument, init_g_attr
from devlib import get_range

LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]


@dataclass
class Batch:
    ipa_matrix: LongTensor
    pos_to_predict: LongTensor
    target_ipa: LongTensor = field(init=False)

    def __post_init__(self):
        batch_i = get_range(self.batch_size, 2, 0)
        window_i = get_range(self.window_size, 2, 1)
        self.target_ipa = ipa_matrix[batch_i, window_i, self.pos_to_predict]

    @property
    def shape(self):
        return self.ipa_matrix.shape

    @property
    def batch_size(self):
        return self.ipa_matrix.size(0)

    @property
    def window_size(self):
        return self.ipa_matrix.size(1)


class IpaDataset(Dataset):

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # FIXME(j_luo) Not sure this is the right way since I don't know what the data actually looks like.
        return self.data.iloc[idx]


def collate_fn():
    # FIXME(j_luo) This needs to be filled out.
    pass


@init_g_attr
class IpaDataLoader(DataLoader):

    add_argument('batch_size', default=16, dtype=int, msg='batch size')
    add_argument('num_workers', default=5, dtype=int, msg='number of workers for the data loader')

    def __init__(self, csv_path: 'a', batch_size, num_workers):
        dataset = IpaDataset(csv_path)
        super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
