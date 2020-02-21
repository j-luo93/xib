from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import ClassVar, Dict, List, Union

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from dev_misc import BT, FT, LT, g
from dev_misc.devlib import BaseBatch, batch_class, get_array, get_length_mask
from dev_misc.trainlib import Task
from dev_misc.trainlib.base_data_loader import BaseDataLoader
from dev_misc.utils import deprecated
from xib.aligned_corpus.char_set import CharSet
from xib.aligned_corpus.corpus import AlignedCorpus
from xib.aligned_corpus.dataset import AlignedDataset, AlignedDatasetItem
from xib.aligned_corpus.transcriber import MultilingualTranscriber
from xib.aligned_corpus.vocabulary import Vocabulary
from xib.batch import BatchSampler, DenseFeatureMatrix, convert_to_dense
from xib.ipa import Category


@batch_class
class BaseAlignedBatch(BaseBatch):
    sentences: np.asarray
    lengths: LT
    source_padding: BT = field(init=False)

    # TODO(j_luo) Get rid of class vars.
    known_vocab: ClassVar[Vocabulary] = None

    def __post_init__(self):
        self.source_padding = ~get_length_mask(self.lengths, self.max_length)
        self.source_padding.rename_('batch', 'length')
        self.lengths.rename_('batch')

    @property
    def batch_size(self) -> int:
        return self.lengths.size(0)

    @property
    def max_length(self) -> int:
        return self.lengths.max().item()


@batch_class
class AlignedIpaBatch(BaseAlignedBatch):
    feat_matrix: LT
    dense_feat_matrix: DenseFeatureMatrix = field(init=False)

    all_lost_dense_feat_matrix: ClassVar[DenseFeatureMatrix] = None

    def __post_init__(self):
        super().__post_init__()
        self.feat_matrix.rename_('batch', 'length', 'feat_group')
        self.dense_feat_matrix = convert_to_dense(self.feat_matrix)


@batch_class
class AlignedTextBatch(BaseAlignedBatch):
    unit_id_seqs: LT = field(init=False)

    lost_char_set: ClassVar[CharSet] = None
    # lost_unit2id: ClassVar[Dict[str, int]] = None

    def __post_init__(self):
        super().__post_init__()

        unit_id_seqs = list()
        for sentence in self.sentences:
            uss = sentence.to_unsegmented(is_known_ipa=True, is_lost_ipa=False, annotated=False)
            unit_id_seq = torch.LongTensor([self.lost_char_set.to_id(char) for char in uss.content])
            unit_id_seqs.append(unit_id_seq)
        self.unit_id_seqs = torch.nn.utils.rnn.pad_sequence(unit_id_seqs, batch_first=True)
        self.unit_id_seqs.rename_('batch', 'length')


AlignedBatch = Union[AlignedIpaBatch, AlignedTextBatch]


def collate_aligned_dataset_items(items: List[AlignedDatasetItem]) -> AlignedBatch:
    sentences = get_array([item.sentence for item in items])
    lengths = default_collate([item.length for item in items])
    if g.input_format == 'ipa':
        feat_matrix = torch.nn.utils.rnn.pad_sequence([item.feat_matrix for item in items], batch_first=True)
        return AlignedIpaBatch(sentences, lengths, feat_matrix)
    else:
        return AlignedTextBatch(sentences, lengths)


class AlignedDataLoader(BaseDataLoader):

    collate_fn = collate_aligned_dataset_items
    dataset: AlignedDataset

    def __init__(self, dataset: AlignedDataset, task: Task, *args, **kwargs):
        is_ipa = g.input_format == 'ipa'
        lengths = [
            sentence.lost_ipa_length if is_ipa else sentence.lost_form_length
            for sentence in dataset.data
        ]
        batch_sampler = BatchSampler([str(d) for d in dataset.data], lengths, shuffle=True, training=task.training)
        super().__init__(dataset, task, *args, batch_sampler=batch_sampler, **kwargs)

        # Assign class variables for AlignedBatch.
        if g.input_format == 'ipa':
            lost_ipa_units = self.dataset.corpus.char_sets[g.lost_lang].id2unit
            fm = torch.cat([ipa_unit.feat_matrix for ipa_unit in lost_ipa_units], dim=0)
            fm = fm.unsqueeze(dim=1).rename('batch', 'length', 'feat')
            dfm = convert_to_dense(fm)
            AlignedIpaBatch.all_lost_dense_feat_matrix = dfm
        else:
            # AlignedTextBatch.lost_unit2id = self.dataset.corpus.unit2id[g.lost_lang]
            AlignedTextBatch.lost_char_set = self.dataset.corpus.char_sets[g.lost_lang]
        # Get vocabulary.
        BaseAlignedBatch.known_vocab = Vocabulary()

    def __iter__(self) -> AlignedBatch:
        for batch in super().__iter__():
            yield batch.cuda()
