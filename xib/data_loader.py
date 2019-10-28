import logging
import random
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial, wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from arglib import add_argument, g, init_g_attr
from devlib import (PandasDataLoader, PandasDataset, dataclass_cuda,
                    dataclass_size_repr, get_length_mask, get_range,
                    get_tensor, get_zeros)
from pycountry import languages
from xib.families import get_all_distances, get_families
from xib.ipa import Category, Index, conditions, should_include

# NOTE(j_luo) Batch dataclasses will inherit the customized __repr__.
batch_class = wraps(partial(dataclass, repr=False))


@batch_class
class BaseBatch:
    segments: np.ndarray
    lengths: torch.LongTensor
    # TODO(j_luo) use the plurals
    feat_matrix: torch.LongTensor
    source_padding: torch.BoolTensor = field(init=False)
    # HACK(j_luo) This is a hack. If we have a way of inheriting names, then this is not necessary.
    batch_name: str = field(default='batch', repr=False)
    length_name: str = field(default='length', repr=False)

    run_post_init: bool = True

    __repr__ = dataclass_size_repr
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
        """Do not override this, override _post_init_helper instead."""
        if self.run_post_init:
            self._post_init_helper()

    def _post_init_helper(self):
        self.feat_matrix = self.feat_matrix.refine_names(self.batch_name, self.length_name, 'feat_group')
        self.source_padding = ~get_length_mask(self.lengths, self.max_length)
        self.source_padding = self.source_padding.refine_names(self.batch_name, self.length_name)
        self.lengths = self.lengths.refine_names(self.batch_name)

    def split(self, size: int):
        """Split the batch into multiple smaller batches with size <= `size`."""
        if size > self.batch_size:
            raise ValueError(f'Size {size} bigger than batch size {self.batch_size}.')
        # Gather all relevant arguments for each smaller batch. Note that some arguments are not part of __init__.
        num_splits = (self.batch_size + size - 1) // size
        inherited_init_kwargs: Dict[str, Tuple[Any, bool]] = dict()
        split_init_kwargs: Dict[str, Tuple[Any, bool]] = dict()
        for attr, field in self.__dataclass_fields__.items():
            anno = field.type
            value = getattr(self, attr)
            init = field.init
            if anno is np.ndarray:
                lengths = [size * i for i in range(1, 1 + num_splits)]
                split_init_kwargs[attr] = (np.split(value, lengths, axis=0), init)
            elif 'Tensor' in anno.__name__:
                value = value.align_to(self.batch_name, ...)
                values = [v.refine_names(*value.names) for v in value.rename(None).split(size, dim=0)]
                split_init_kwargs[attr] = (values, init)
            else:
                inherited_init_kwargs[attr] = (value, init)

        batches = list()
        batch_cls = type(self)
        for i in range(num_splits):
            init_kwargs = dict()
            remaining_fields = dict()
            for attr, (value, init) in inherited_init_kwargs.items():
                d = init_kwargs if init else remaining_fields
                d[attr] = value
            for attr, (value, init) in split_init_kwargs.items():
                d = init_kwargs if init else remaining_fields
                d[attr] = value[i]
            init_kwargs['run_post_init'] = False
            batch = batch_cls(**init_kwargs)
            for attr, value in remaining_fields.items():
                setattr(batch, attr, value)
            batches.append(batch)
        return batches


@batch_class
class IpaBatch(BaseBatch):
    pos_to_predict: torch.LongTensor = field(init=False)
    target_feat: torch.LongTensor = field(init=False)
    target_weight: torch.FloatTensor = field(init=False)

    _g2f = None

    def _post_init_helper(self):
        bs, ml, nfg = self.feat_matrix.shape
        new_bs = bs * ml
        batch_i = get_range(bs, 2, 0, cpu=True)

        self.segments = np.repeat(self.segments, ml)
        self.target_weight = get_length_mask(self.lengths, ml, cpu=True)
        self.target_weight = self.target_weight.unsqueeze(dim=-1).repeat(1, 1, nfg).view(new_bs, nfg).float()
        self.pos_to_predict = get_range(ml, 2, 1, cpu=True).repeat(bs, 1)

        self.lengths = self.lengths.repeat_interleave(ml, dim=0)

        # NOTE(j_luo) This is global index.
        target_feat = self.feat_matrix[batch_i, self.pos_to_predict].view(new_bs, -1)
        self.pos_to_predict = self.pos_to_predict.view(new_bs)
        self.feat_matrix = self.feat_matrix.repeat_interleave(ml, dim=0)
        # Get conversion matrix.
        if self._g2f is None:
            total = Index.total_indices()
            self._g2f = torch.LongTensor(total)
            indices = [Index.get_feature(i).value for i in range(total)]
            for index in indices:
                self._g2f[index.g_idx] = index.f_idx
        # NOTE(j_luo) This is feature index.
        self.target_feat = self._g2f[target_feat]

        # NOTE(j_luo) If the condition is not satisfied, the target weight should be set to 0.
        for cat, index in conditions.items():
            idx = cat.value
            condition_idx = index.f_idx
            mask = condition_idx != self.target_feat[:, index.c_idx]
            self.target_weight[mask, idx] = 0.0

        # NOTE(j_luo) Refine names.
        # IDEA(j_luo) We can move this process a bit earlier to DataLoader (serialization not yet implemented for named tensors).
        self.pos_to_predict = self.pos_to_predict.refine_names(self.batch_name)
        self.target_feat = self.target_feat.refine_names(self.batch_name, 'feat_group')
        self.target_weight = self.target_weight.refine_names(self.batch_name, 'feat_group')

        super()._post_init_helper()

    def __len__(self):
        return self.batch_size


@batch_class
class SparseIpaBatch(IpaBatch):
    feat_matrix: Dict[Category, torch.FloatTensor]

    def _post_init_helper(self):
        super()._post_init_helper()
        names = self.feat_matrix.names
        bs = self.feat_matrix.size('batch')
        ml = self.feat_matrix.size('length')
        fm = self._g2f[self.feat_matrix.rename(None)].refine_names(*names)
        sfms = dict()
        for cat in Category:
            sfm_idx = fm[..., cat.value]
            sfm = get_zeros(bs, ml, len(cat))
            sfm = sfm.scatter(2, sfm_idx.rename(None).unsqueeze(dim=-1), 1.0)
            sfms[cat] = sfm
        self.feat_matrix = sfms


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


def collate_fn(batch) -> CollateReturn:
    segments = list()
    matrices = list()
    lengths = list()
    for segment, matrix in batch:
        length = len(matrix)
        segments.append(segment)
        lengths.append(length)
        matrices.append(matrix)
    matrices = torch.nn.utils.rnn.pad_sequence(matrices, batch_first=True)
    segments = np.asarray(segments)
    lengths = torch.LongTensor(lengths)
    return CollateReturn(segments, lengths, matrices)


@init_g_attr
class BatchSampler(Sampler):

    def __init__(self, dataset: 'a', char_per_batch: 'p', shuffle: 'p' = True):
        self.dataset = dataset
        # Partition the entire dataset beforehand into batches by length.
        lengths = np.asarray(list(map(len, self.dataset.data['matrices'])))
        indices = lengths.argsort()[::-1]  # NOTE(j_luo) Sort in descending order.
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


@init_g_attr
class BaseIpaDataLoader(DataLoader, metaclass=ABCMeta):

    add_argument('data_path', dtype='path', msg='path to the feat data in tsv format.')
    add_argument('num_workers', default=5, dtype=int, msg='number of workers for the data loader')
    add_argument('char_per_batch', default=500, dtype=int, msg='batch_size')

    def __init__(self, data_path: 'p', char_per_batch: 'p', num_workers):
        dataset = IpaDataset(data_path)
        batch_sampler = BatchSampler(dataset, char_per_batch, shuffle=True)
        cls = type(self)
        super().__init__(dataset, batch_sampler=batch_sampler,
                         num_workers=num_workers, collate_fn=collate_fn)

    @abstractmethod
    def _prepare_batch(self):
        pass

    def __iter__(self):
        for collate_return in super().__iter__():
            batch = self._prepare_batch(collate_return)
            yield batch


class IpaDataLoader(BaseIpaDataLoader):

    batch_cls: Type[BaseBatch] = IpaBatch

    def _prepare_batch(self, collate_return: CollateReturn) -> IpaBatch:
        cls = type(self)
        batch_cls = cls.batch_cls
        return batch_cls(collate_return.segments, collate_return.lengths, collate_return.matrices).cuda()


class SparseIpaDataLoader(IpaDataLoader):
    batch_cls = SparseIpaBatch


@batch_class
class ContinuousTextIpaBatch(BaseBatch):
    pass


class ContinuousTextDataLoader(IpaDataLoader):

    def _prepare_batch(self, collate_return: CollateReturn) -> ContinuousTextIpaBatch:
        return ContinuousTextIpaBatch(collate_return.segments, collate_return.lengths, collate_return.matrices)

# ------------------------------------------------------------- #
#                         Metric learner                        #
# ------------------------------------------------------------- #


@dataclass
class MetricLearningBatch:
    lang1: np.ndarray
    lang2: np.ndarray
    normalized_score: torch.FloatTensor
    dist: torch.FloatTensor

    cuda = dataclass_cuda

    def __post_init__(self):
        self.normalized_score = get_tensor(self.normalized_score)  # .refine_names('batch', 'feat_group')
        self.dist = get_tensor(self.dist)  # .refine_names('batch')

    def __len__(self):
        return self.dist.size(0)


def _get_metric_data(data_path: Path, emb_groups: str, family_file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(data_path, sep='\t')
    data = pd.pivot_table(data, index=['lang1', 'lang2'], columns='category', values='normalized_score').reset_index()
    cats = [cat.name for cat in Category if should_include(emb_groups, cat)] + ['avg']
    cols = ['lang1', 'lang2'] + cats
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
    return data


@init_g_attr
class MetricLearningDataLoader(PandasDataLoader):

    add_argument('family_file_path', dtype='path', msg='path to the family file')
    add_argument('num_lang_pairs', dtype=int, default=10, msg='number of languages')

    def __init__(self, data_path, num_workers, emb_groups: 'p', family_file_path: 'p', num_lang_pairs: 'p', data=None):
        if data is None:
            data = _get_metric_data(data_path, emb_groups, family_file_path)
        self.all_langs = sorted(set(data['lang1']))
        self.cats = [cat.name for cat in Category if should_include(emb_groups, cat)] + ['avg']
        super().__init__(data, batch_size=num_lang_pairs, num_workers=num_workers)

    def __iter__(self) -> MetricLearningBatch:
        for df in super().__iter__():
            lang1 = df['lang1'].values
            lang2 = df['lang2'].values
            normalized_score = get_tensor(df[self.cats].values.astype('float32'))
            dist = get_tensor(df['dist'].values.astype('float32'))
            return MetricLearningBatch(lang1, lang2, normalized_score, dist).cuda()

    def select(self, langs1: Sequence[str], langs2: Sequence[str]) -> 'MetricLearningDataLoader':
        all_langs1 = set(langs1)
        all_langs2 = set(langs2)
        data = self.dataset.data
        mask = (data['lang1'].isin(all_langs1)) & (data['lang2'].isin(all_langs2))
        data = data[mask].reset_index(drop=True)
        return MetricLearningDataLoader(data=data)
