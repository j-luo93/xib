import logging
import random
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from pycountry import languages
from torch.utils.data import DataLoader, Dataset, Sampler

from arglib import add_argument, g, init_g_attr
from devlib import (PandasDataLoader, dataclass_cuda, dataclass_size_repr,
                    get_length_mask, get_range, get_tensor)
from xib.families import get_all_distances, get_families
from xib.ipa import Category, Index, conditions, should_include


@dataclass
class BaseBatch:
    segments: np.ndarray
    lengths: torch.LongTensor
    # TODO(j_luo) use the plurals
    feat_matrix: torch.LongTensor
    source_padding: torch.BoolTensor = field(init=False)

    cuda = dataclass_cuda

    @property
    def shape(self):
        return self.feat_matrix.shape

    @property
    def batch_size(self):
        return self.feat_matrix.size(0)

    @property
    def max_length(self):
        return self.feat_matrix.size(1)

    def __post_init__(self):
        self.lengths = self.lengths.refine_names('batch')
        self.feat_matrix = self.feat_matrix.refine_names('batch', 'length', 'feat_group')
        self.source_padding = ~get_length_mask(self.lengths, self.max_length)
        self.source_padding = self.source_padding.refine_names('batch', 'length')
        self.cuda()


@dataclass
class IpaBatch(BaseBatch):
    pos_to_predict: torch.LongTensor
    target_feat: torch.LongTensor = field(init=False)
    target_weight: torch.FloatTensor = field(init=False)

    _g2f = None

    __repr__ = dataclass_size_repr

    def __post_init__(self):
        batch_i = get_range(self.batch_size, 1, 0)
        # NOTE(j_luo) This is global index. # TODO(j_luo) ugly
        target_feat = self.feat_matrix.rename(None)[batch_i, self.pos_to_predict]
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
        # IDEA(j_luo) We can move this process a bit earlier to DataLoader (serialization not yet implemented for named tensors).
        self.pos_to_predict = self.pos_to_predict.refine_names('batch')
        self.target_feat = self.target_feat.refine_names('batch', 'feat_group')
        self.target_weight = self.target_weight.refine_names('batch', 'feat_group')

        super().__post_init__()

    def __len__(self):
        return self.batch_size


class IpaDataset(Dataset):

    def __init__(self, data_path):
        self.data = torch.load(data_path)
        logging.info(f'Loaded {len(self)} segments in total.')

    def __len__(self):
        return len(self.data['segments'])

    def __getitem__(self, idx):
        return self.data['segments'][idx], self.data['matrices'][idx]


@dataclass
class CollateReturn:
    segments: np.ndarray
    lengths: torch.LongTensor
    matrices: torch.LongTensor
    positions: torch.LongTensor = None


def _collate_fn(batch, repeat=True) -> CollateReturn:
    segments = list()
    matrices = list()
    lengths = list()
    if repeat:
        positions = list()
    for segment, matrix in batch:
        length = len(matrix)
        if repeat:
            segments.extend([segment] * length)
            lengths.extend([length] * length)
            matrices.extend([matrix] * length)
            positions.extend(list(range(length)))
        else:
            segments.append(segment)
            lengths.append(length)
            matrices.append(matrix)
    matrices = torch.nn.utils.rnn.pad_sequence(matrices, batch_first=True)
    segments = np.asarray(segments)
    lengths = torch.LongTensor(lengths)
    if repeat:
        positions = torch.LongTensor(positions)
        return CollateReturn(segments, lengths, matrices, positions=positions)
    else:
        return CollateReturn(segments, lengths, matrices)


def lm_collate_fn(batch):
    return _collate_fn(batch, repeat=True)


def decipher_collate_fn(batch):
    return _collate_fn(batch, repeat=False)


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


@init_g_attr
class BaseIpaDataLoader(DataLoader, metaclass=ABCMeta):

    collate_fn = None

    add_argument('data_path', dtype=str, msg='path to the feat data in tsv format.')
    add_argument('num_workers', default=5, dtype=int, msg='number of workers for the data loader')
    add_argument('char_per_batch', default=500, dtype=int, msg='batch_size')

    def __init__(self, data_path: 'p', char_per_batch: 'p', num_workers):
        dataset = IpaDataset(data_path)
        batch_sampler = BatchSampler(dataset, char_per_batch, shuffle=True)
        cls = type(self)
        super().__init__(dataset, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=cls.collate_fn)

    @abstractmethod
    def _prepare_batch(self):
        pass

    def __iter__(self):
        for collate_return in super().__iter__():
            batch = self._prepare_batch(collate_return)
            yield batch


class IpaDataLoader(BaseIpaDataLoader):

    collate_fn = lm_collate_fn

    def _prepare_batch(self, collate_return: CollateReturn) -> IpaBatch:
        return IpaBatch(collate_return.segments, collate_return.lengths, collate_return.matrices, collate_return.positions)


@dataclass
class ContinuousTextIpaBatch(BaseBatch):
    pass


class ContinuousTextDataLoader(IpaDataLoader):

    collate_fn = decipher_collate_fn

    def _prepare_batch(self, collate_return: CollateReturn) -> ContinuousTextIpaBatch:
        return ContinuousTextIpaBatch(collate_return.segments, collate_return.lengths, collate_return.matrices)


@dataclass
class MetricLearningBatch:
    lang1: np.ndarray
    lang2: np.ndarray
    normalized_score: torch.FloatTensor
    dist: torch.FloatTensor

    cuda = dataclass_cuda

    def __post_init__(self):
        self.normalized_score = get_tensor(self.normalized_score).refine_names('batch', 'feat_group')
        self.dist = get_tensor(self.dist).refine_names('batch')
        self.cuda()

    def __len__(self):
        return self.dist.size('batch')


@init_g_attr()
class MetricLearningDataLoader(PandasDataLoader):

    add_argument('family_file_path', dtype=str, msg='path to the family file')
    add_argument('num_lang_pairs', dtype=int, default=10, msg='number of languages')

    def __init__(self, data_path: 'p', num_workers, emb_groups: 'p', family_file_path: 'p', num_lang_pairs: 'p'):
        # Get scores first.
        data = pd.read_csv(data_path, sep='\t')
        data = pd.pivot_table(data, index=['lang1', 'lang2'], columns='category',
                              values='normalized_score').reset_index()
        self.cats = [cat.name for cat in Category if should_include(emb_groups, cat)] + ['avg']
        cols = ['lang1', 'lang2'] + self.cats
        data = data[cols]

        # Get ground truth distances.
        get_families(family_file_path)
        dists = get_all_distances()

        def _get_lang(lang: str):
            if len(lang) == 2:
                return languages.get(alpha_2=lang)
            elif len(lang) == 3:
                return languages.get(alpha_3=lang)
            else:
                return None

        def _get_dist(lang1: str, lang2: str):
            lang_struct1 = _get_lang(lang1)
            lang_struct2 = _get_lang(lang2)
            if lang_struct1 is None or lang_struct2 is None:
                return None
            return dists.get((lang_struct1.name, lang_struct2.name), None)

        dists = [_get_dist(lang1, lang2) for lang1, lang2, *_ in data.values]
        data['dist'] = dists
        cols.append('dist')
        data = data[~data['dist'].isnull()].reset_index(drop=True)

        self.all_langs = sorted(set(data['lang1']))
        super().__init__(data, columns=cols, batch_size=num_lang_pairs, num_workers=num_workers)

    def __iter__(self) -> MetricLearningBatch:
        for df in super().__iter__():
            lang1 = df['lang1'].values
            lang2 = df['lang2'].values
            normalized_score = get_tensor(df[self.cats].values.astype('float32'))
            dist = get_tensor(df['dist'].values.astype('float32'))
            yield MetricLearningBatch(lang1, lang2, normalized_score, dist)

    def select(self, langs: Sequence[str]):
        all_langs = set(langs)
        mask = (self.dataset.data['lang1'].isin(all_langs)) & (self.dataset.data['lang2'].isin(all_langs))
        self.dataset.select(mask)
