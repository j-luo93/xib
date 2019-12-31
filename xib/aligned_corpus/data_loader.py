from dataclasses import field
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from dev_misc import BT, FT, LT
from dev_misc.devlib import BaseBatch, batch_class, get_array, get_length_mask
from dev_misc.trainlib.base_data_loader import BaseDataLoader
from dev_misc.utils import deprecated
from xib.aligned_corpus.dataset import AlignedDatasetItem
from xib.batch import DenseFeatureMatrix, convert_to_dense
from xib.ipa import Category


@batch_class
class AlignedBatch(BaseBatch):
    sentences: np.asarray
    lengths: LT
    feat_matrix: LT
    dense_feat_matrix: DenseFeatureMatrix = field(init=False)
    source_padding: BT = field(init=False)

    def __post_init__(self):
        self.dense_feat_matrix = convert_to_dense(self.feat_matrix)
        self.source_padding = ~get_length_mask(self.lengths, self.max_length)

    @property
    def batch_size(self) -> int:
        return self.lengths.size(0)

    @property
    def max_length(self) -> int:
        return self.lengths.size(1)


def collate_aligned_dataset_items(items: List[AlignedDatasetItem]) -> AlignedBatch:
    sentences = get_array([item.sentence for item in items])
    lengths = default_collate([item.length for item in items])
    feat_matrix = torch.nn.utils.rnn.pad_sequence([item.feat_matrix for item in items], batch_first=True)
    return AlignedBatch(sentences, lengths, feat_matrix)


class AlignedDataLoader(BaseDataLoader):

    collate_fn = collate_aligned_dataset_items

    def __iter__(self) -> AlignedBatch:
        for batch in super().__iter__():
            return batch.cuda()
