from dataclasses import field
from typing import Any, Dict, Tuple

import numpy as np
import torch

from dev_misc import FT, LT, g
from dev_misc.devlib import BaseBatch as BaseBatchDev
from dev_misc.devlib import (batch_class, get_array, get_length_mask,
                             get_range, get_zeros)
from xib.ipa import Category, Index, conditions, get_enum_by_cat


@batch_class
class BaseBatch(BaseBatchDev):
    segments: np.ndarray
    lengths: torch.LongTensor
    feat_matrix: torch.LongTensor
    source_padding: torch.BoolTensor = field(init=False)
    # TODO(j_luo) This is a hack. If we have a way of inheriting names, then this is not necessary.
    batch_name: str = field(default='batch', repr=False)
    length_name: str = field(default='length', repr=False)

    run_post_init: bool = True

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
        # IDEA(j_luo) Use fields function instead.
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


def mask_out_target_weight(target_weight: FT, target_feat: LT):
    for cat, index in conditions.items():
        idx = cat.value
        condition_idx = index.f_idx
        mask = condition_idx != target_feat[:, index.c_idx]
        target_weight[mask, idx] = 0.0


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
        mask_out_target_weight(self.target_weight, self.target_feat)

        # NOTE(j_luo) Refine names.
        # TODO(j_luo) We can move this process a bit earlier to DataLoader (serialization not yet implemented for named tensors).
        self.pos_to_predict = self.pos_to_predict.refine_names(self.batch_name)
        self.target_feat = self.target_feat.refine_names(self.batch_name, 'feat_group')
        self.target_weight = self.target_weight.refine_names(self.batch_name, 'feat_group')

        super()._post_init_helper()

    def __len__(self):
        return self.batch_size


@batch_class
class CbowIpaBatch(IpaBatch):

    def _post_init_helper(self):
        bs, ml, nfg = self.feat_matrix.shape

        self.target_weight = get_zeros(bs, cpu=True).float().unsqueeze(dim=-1).repeat(1, nfg).fill_(1.0)
        # self.target_weight = self.target_weight.unsqueeze(dim=-1).repeat(1, 1, nfg).float()
        self.pos_to_predict = get_zeros(bs, cpu=True).long().fill_(g.window_size // 2)

        # NOTE(j_luo) This is global index.
        target_feat = self.feat_matrix[:, g.window_size // 2]

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
        mask_out_target_weight(self.target_weight, self.target_feat)

        # NOTE(j_luo) Refine names.
        self.pos_to_predict = self.pos_to_predict.refine_names(self.batch_name)
        self.target_feat = self.target_feat.refine_names(self.batch_name, 'feat_group')
        self.target_weight = self.target_weight.refine_names(self.batch_name, 'feat_group')

        BaseBatch._post_init_helper(self)


@batch_class
class DenseIpaBatch(IpaBatch):
    dense_feat_matrix: Dict[Category, torch.FloatTensor] = field(init=False)

    def _post_init_helper(self):
        super()._post_init_helper()
        names = self.feat_matrix.names
        bs = self.feat_matrix.size('batch')
        ml = self.feat_matrix.size('length')
        fm = self._g2f[self.feat_matrix.rename(None)].refine_names(*names)
        sfms = dict()
        for cat in Category:
            e = get_enum_by_cat(cat)
            sfm_idx = fm[..., cat.value]
            sfm = get_zeros(bs, ml, len(e), cpu=True)
            sfm = sfm.scatter(2, sfm_idx.rename(None).unsqueeze(dim=-1), 1.0)
            sfms[cat] = sfm.refine_names('batch', 'length', f'{cat.name}_feat')
        self.dense_feat_matrix = {k: v.cuda() for k, v in sfms.items()}
