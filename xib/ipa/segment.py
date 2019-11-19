from abc import ABC, abstractmethod
from typing import List

import torch

import xib.ipa.process as process
from dev_misc.devlib import BT, LT
from dev_misc.utils import cached_property

B, I, O = 0, 1, 2

class BaseSegment(ABC):

    @property
    @abstractmethod
    def gold_tag_seq(self) -> LT:
        pass

    @cached_property
    @abstractmethod
    def feat_matrix(self) -> LT:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        cls = type(self)
        return f'{cls.__name__}("{self}")'


class Segment(BaseSegment):

    def __init__(self, token: str):
        self.ipa = process.get_string(token)
        self.token = token
        self._merged = False
        if len(self.ipa) == 0:
            raise ValueError('Invalid IPA string.')
        self._apply_all()
        self._merge()
        self._indexify()

    def __len__(self):
        return len(self.feat_matrix)

    @property
    def gold_tag_seq(self) -> LT:
        return torch.LongTensor([B] + [I] * (len(self) - 1))

    def __str__(self):
        return '-'.join(''.join(map(str, unit)) for unit in self.merged_ipa)

    def _apply_all(self):
        for name, dg in process.name2dg.items():
            setattr(self, name, process.get_dg_value(self.ipa, dg))
        if self.ptype[0] not in ['consonant', 'vowel']:
            raise ValueError('Invalid IPA string.')

    def __getitem__(self, feat: str):
        if self._merged:
            try:
                return self.datum_cols[feat]
            except KeyError:
                return self.datum_inds[feat]
        else:
            try:
                return getattr(self, feat)
            except AttributeError:
                raise KeyError(f'Key {feat} not found.')

    def _merge(self):
        datum = process.merge_ipa(self, self.ipa, self.token)
        if not datum:
            raise ValueError('Invalid IPA string.')
        self.merged_ipa = datum[2]
        self.datum_cols = {
            feat: datum[3 + i]
            for i, feat in enumerate(process.normal_feats + process.feats_to_merge)
        }
        self._merged = True

    def _indexify(self):
        self.datum_inds = {
            f'{feat}_idx': process.indexify_ipa(feat, value)
            for feat, value in self.datum_cols.items()
        }

    @cached_property
    def feat_matrix(self) -> LT:
        return process.get_feat_matrix(self)


class SegmentWindow(BaseSegment):

    def __init__(self, segments: List[Segment]):
        self._segments = segments

    def __len__(self):
        return sum(len(segment) for segment in self._segments)

    @property
    def gold_tag_seq(self) -> LT:
        return torch.cat([segment.gold_tag_seq for segment in self._segments], dim=0)

    @cached_property
    def feat_matrix(self) -> LT:
        matrices = [segment.feat_matrix for segment in self._segments]
        return torch.cat(matrices, dim=0)

    def __str__(self):
        return ' '.join(str(segment) for segment in self._segments)
