from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import ClassVar, Dict, List

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from dev_misc import BT, FT, LT, g
from dev_misc.devlib import BaseBatch, batch_class, get_array, get_length_mask
from dev_misc.trainlib import Task
from dev_misc.trainlib.base_data_loader import BaseDataLoader
from dev_misc.utils import deprecated
from xib.aligned_corpus.corpus import AlignedCorpus, Vocabulary
from xib.aligned_corpus.dataset import AlignedDataset, AlignedDatasetItem
from xib.aligned_corpus.transcriber import MultilingualTranscriber
from xib.batch import DenseFeatureMatrix, convert_to_dense
from xib.ipa import Category


@batch_class
class AlignedBatch(BaseBatch):
    sentences: np.asarray
    lengths: LT
    feat_matrix: LT
    dense_feat_matrix: DenseFeatureMatrix = field(init=False)
    source_padding: BT = field(init=False)

    all_lost_dense_feat_matrix: ClassVar[DenseFeatureMatrix] = None
    known_vocab: ClassVar[Vocabulary] = None

    def __post_init__(self):
        self.source_padding = ~get_length_mask(self.lengths, self.max_length)
        self.source_padding.rename_('batch', 'length')
        self.lengths.rename_('batch')

        self.feat_matrix.rename_('batch', 'length', 'feat_group')
        self.dense_feat_matrix = convert_to_dense(self.feat_matrix)

    @property
    def batch_size(self) -> int:
        return self.lengths.size(0)

    @property
    def max_length(self) -> int:
        return self.lengths.max().item()


def collate_aligned_dataset_items(items: List[AlignedDatasetItem]) -> AlignedBatch:
    sentences = get_array([item.sentence for item in items])
    lengths = default_collate([item.length for item in items])
    feat_matrix = torch.nn.utils.rnn.pad_sequence([item.feat_matrix for item in items], batch_first=True)
    return AlignedBatch(sentences, lengths, feat_matrix)


class AlignedDataLoader(BaseDataLoader):

    collate_fn = collate_aligned_dataset_items
    dataset: AlignedDataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Assign class variables for AlignedBatch.
        lost_ipa_units = self.dataset.corpus.id2unit[g.lost_lang]
        fm = torch.cat([ipa_unit.feat_matrix for ipa_unit in lost_ipa_units], dim=0)
        fm = fm.unsqueeze(dim=1).rename('batch', 'length', 'feat')
        dfm = convert_to_dense(fm)
        AlignedBatch.all_lost_dense_feat_matrix = dfm
        # Get vocabulary.
        AlignedBatch.known_vocab = Vocabulary()

    def __iter__(self) -> AlignedBatch:
        for batch in super().__iter__():
            yield batch.cuda()
