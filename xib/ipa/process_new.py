from __future__ import annotations

import logging
import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (Callable, ClassVar, Iterable, Iterator, List, Optional,
                    Sequence, TextIO, Tuple, TypeVar, Union)

import numpy as np
import pandas as pd
import torch
from ipapy import UNICODE_TO_IPA
from ipapy.ipachar import (DG_C_MANNER, DG_C_PLACE, DG_C_VOICING,
                           DG_DIACRITICS, DG_S_BREAK, DG_S_LENGTH, DG_S_STRESS,
                           DG_T_CONTOUR, DG_T_GLOBAL, DG_T_LEVEL, DG_TYPES,
                           DG_V_BACKNESS, DG_V_HEIGHT, DG_V_ROUNDNESS)
from ipapy.ipastring import IPAChar, IPAString
from tqdm import tqdm

from dev_misc import add_argument, g
from dev_misc.devlib import BT, LT
from dev_misc.utils import cached_property, deprecated
from xib.ipa import Category


to_remove_rules = {
    'n͡m': ['n', 'm'],
    'd͡b': ['d', 'b']
}

to_remove = {
    k: [UNICODE_TO_IPA[v] for v in vs]
    for k, vs in to_remove_rules.items()
}

B, I, O = 0, 1, 2


def get_string(s: str) -> IPAString:
    ipa_chars = list()
    for c in s:
        c_str = str(c)
        if c_str in to_remove:
            ipa_chars.extend(to_remove[c_str])
        else:
            ipa_chars.append(c)
        if g.use_atomic_ipa:
            ipas = list()
            for c in s:
                ipas.append(IPAString(unicode_string=c))
    return IPAString(ipa_chars)


def get_dg_value(s: IPAString, dg) -> List:
    return [c.dg_value(dg) for c in s.ipa_chars]


name2dg = {
    'ptype': DG_TYPES,
    'c_voicing': DG_C_VOICING,
    'c_place': DG_C_PLACE,
    'c_manner': DG_C_MANNER,
    'v_height': DG_V_HEIGHT,
    'v_backness': DG_V_BACKNESS,
    'v_roundness': DG_V_ROUNDNESS,
    'diacritics': DG_DIACRITICS,
    's_stress': DG_S_STRESS,
    's_length': DG_S_LENGTH,
    's_break': DG_S_BREAK,
    't_level': DG_T_LEVEL,
    't_contour': DG_T_CONTOUR,
    't_global': DG_T_GLOBAL
}


@dataclass
class Span:
    value: str
    start: int
    end: int

    @deprecated
    def __eq__(self, other: Span):
        return self.is_same_span(other)

    def is_same_span(self, other: Span) -> bool:
        if not isinstance(other, Span):
            return False
        return self.start == other.start and self.end == other.end

    def is_same_word(self, other: Span) -> bool:
        return self.is_same_span(other) and self.plain_value == other.plain_value

    def __str__(self):
        return f'{self.value}:{self.start}:{self.end}'

    def __len__(self):
        return self.end - self.start + 1

    @cached_property
    def plain_value(self) -> str:
        return re.sub(r'[-\]\[#]', '', self.value)

    @deprecated
    def is_prefix_of(self, other: Span) -> bool:
        return self.is_prefix_span_of(other)

    def is_prefix_span_of(self, other: Span) -> bool:
        if not isinstance(other, Span):
            return False
        return self.start == other.start and self.end <= other.end and len(self) > len(other) * 0.5

    def is_prefix_word_of(self, other: Span) -> bool:
        return self.is_prefix_span_of(other) and other.plain_value.startswith(self.plain_value) and len(self) > len(other) * 0.5


@dataclass
class Segmentation:
    spans: List[Span]

    def __len__(self):
        return len(self.spans)

    def __iter__(self) -> Iterator[Span]:
        yield from self.spans

    def __str__(self):
        return ' '.join([str(span) for span in self.spans])


class BaseSegment(ABC):

    has_gold_tag_seq: ClassVar[bool]

    @property
    @abstractmethod
    def feat_matrix(self) -> LT: ...

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def __str__(self): ...

    def __repr__(self):
        cls = type(self)
        return f'{cls.__name__}("{self}")'

    @abstractmethod
    def __getitem__(self, idx: int) -> str:
        """Get the corresponding unit (merged) given the index."""

    @property
    @abstractmethod
    def segment_list(self) -> List[str]:
        """Represent a list of IPAString, as a list of units."""

    @property
    @abstractmethod
    def merged_ipa(self) -> List[IPAString]:
        """Return a list of IPAString."""

    @cached_property
    def cv_list(self) -> List[str]:
        """Return a list of strings corresponding to the consonants and vowels."""
        ret = list()
        for ipa_unit in self.merged_ipa:
            unit = list()
            for c in ipa_unit:
                if c.is_vowel or c.is_consonant:
                    unit.append(str(c))
            if not unit:
                raise ValueError(f'There is no consonant/vowel in this unit.')
            ret.append(''.join(unit))
        return ret

    def __eq__(self, other: BaseSegment):
        if not isinstance(other, BaseSegment):
            return False
        else:
            return self.segment_list == other.segment_list

    def __hash__(self):
        return hash(tuple(self.segment_list))


class BaseSegmentWithGoldTagSeq(BaseSegment):

    has_gold_tag_seq: ClassVar[bool] = True

    @property
    @abstractmethod
    def gold_tag_seq(self) -> LT: ...


add_argument('min_word_length', default=4, dtype=int, msg='Min length of words.')


class Segment(BaseSegmentWithGoldTagSeq):

    def __init__(self, raw_token: str):
        self._raw_token = raw_token
        self.is_noise = raw_token.startswith('#')
        self.token = raw_token[1:] if self.is_noise else raw_token
        self.ipa = get_string(self.token)
        self._merged = False
        if len(self.ipa) == 0:
            raise ValueError('Invalid IPA string.')
        self._apply_all()
        self._merge()
        self._indexify()

    @property
    def merged_ipa(self):
        return self._merged_ipa

    def __len__(self):
        return len(self._merged_ipa)

    @property
    def gold_tag_seq(self) -> LT:
        if self.is_noise or len(self) < g.min_word_length or len(self) > g.max_word_length:
            return torch.LongTensor([O] * len(self))
        else:
            return torch.LongTensor([B] + [I] * (len(self) - 1))

    @property
    def segment_list(self) -> List[str]:
        return [''.join(map(str, unit)) for unit in self._merged_ipa]

    def permute(self) -> str:
        return ''.join(random.sample(self.segment_list, len(self)))

    def __str__(self):
        return '#' * self.is_noise + '-'.join(self.segment_list)

    def _apply_all(self):
        for name, dg in name2dg.items():
            setattr(self, name, get_dg_value(self.ipa, dg))
        if self.ptype[0] not in ['consonant', 'vowel']:  # pylint: disable=no-member
            raise ValueError('Invalid IPA string.')

    def __getitem__(self, feat_or_idx: Union[int, str]):
        if isinstance(feat_or_idx, str):
            return self._legacy_getitem(feat_or_idx)
        else:
            return self.segment_list[feat_or_idx]

    def _legacy_getitem(self, feat: str):
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
        datum = merge_ipa(self, self.ipa, self.token)
        if not datum:
            raise ValueError('Invalid IPA string.')
        self._merged_ipa = datum[2]
        self.datum_cols = {
            feat: datum[3 + i]
            for i, feat in enumerate(normal_feats + feats_to_merge)
        }
        self._merged = True

    def _indexify(self):
        self.datum_inds = {
            f'{feat}_idx': indexify_ipa(feat, value)
            for feat, value in self.datum_cols.items()
        }

    @cached_property
    def feat_matrix(self) -> LT:
        ret = get_feat_matrix(self)
        if len(ret) != len(self):
            raise RuntimeError(f'Feature matrix has a different length from merged_ipa.')
        return ret

    def to_span(self) -> Optional[Span]:
        if not self.is_noise and len(self) >= g.min_word_length and len(self) <= g.max_word_length:
            span = Span(str(self), 0, len(self) - 1)
            return span

    def break_segment(self, start: int, end: int) -> Union[BrokenSegment, Segment]:
        if start == 0 and end == len(self) - 1:
            return self

        new_feat_matrix = self.feat_matrix[start: end + 1]
        new_list_of_units = self.segment_list[start: end + 1]
        new_list_of_ipas = self.merged_ipa[start: end + 1]
        new_gold_tag_seq = torch.LongTensor([O] * (end + 1 - start))
        return BrokenSegment(new_list_of_units, new_list_of_ipas, new_feat_matrix, new_gold_tag_seq, self)


S = TypeVar('S', Segment, BrokenSegment, SegmentX)


def _apply(series: pd.Series, func: Callable[..., None], progress: bool = False):
    progress_func = series.progress_apply if progress else series.apply
    return progress_func(func)


def apply(df, dg, col_name, progress=False):
    df[col_name] = _apply(df['ipa'], lambda s: get_dg_value(s, dg), progress=progress)


def de_none(s):
    return 'none' if s is None else s


Source = Iterator[str]


def get_ipa_data(source: Source, progress=False) -> Tuple[int, int, pd.DataFrame]:
    cnt = 0
    total = 0
    data = list()
    iterator = iter(source)
    if progress:
        iterator = tqdm(iterator)
    for token in iterator:
        token = token.strip()
        try:
            segment = Segment(token)
            data.append((segment.token, segment.ipa))
        except ValueError:
            cnt += 1
        total += 1

    df = pd.DataFrame(data, columns=['segment', 'ipa'])
    return cnt, total, df


def apply_all(df, progress=False):
    for name, dg in name2dg.items():
        apply(df, dg, name, progress=progress)


def clean_data(df, progress=False):
    len_mask = (df['ipa'].str.len() > 0)
    clean_df = df[len_mask]

    # Some segments do not start with consonants or vowels.
    mask = _apply(clean_df['ptype'], lambda l: l[0] in ['consonant', 'vowel'], progress=progress)
    clean_df = clean_df[mask]
    return clean_df


normal_feats = ['ptype', 'c_voicing', 'c_place', 'c_manner', 'v_height', 'v_backness', 'v_roundness']
feats_to_merge = ['diacritics', 's_stress', 's_length', 's_break', 't_level', 't_contour', 't_global']


def merge_ipa(s: Union[pd.Series, Segment], ipa: IPAString, segment: str) -> List:
    i = 0
    keep = True
    datum_cols = {feat: list() for feat in normal_feats + feats_to_merge}
    merged_ipa = list()
    ptypes = s['ptype']
    errors = defaultdict(list)
    while i < len(ptypes):
        # Get ptype and normal features first.
        for feat in normal_feats:
            datum_cols[feat].append(de_none(s[feat][i]))

        # Try to merge characters if needed.
        j = i + 1
        datum_c_to_merge = dict.fromkeys(feats_to_merge)
        while j < len(ptypes) and ptypes[j] not in ['consonant', 'vowel']:
            # Attach j-th char to i-th.
            for feat in feats_to_merge:
                value = s[feat][j]
                if value is not None:
                    try:
                        assert datum_c_to_merge[feat] is None
                        datum_c_to_merge[feat] = value
                    except:
                        errors[feat].append(s)
                        keep = False
            j += 1
        merged_ipa.append(ipa[i:j])
        i = j
        for feat in feats_to_merge:
            datum_cols[feat].append(de_none(datum_c_to_merge[feat]))
    datum = [segment, ipa, merged_ipa] + [datum_cols[feat] for feat in normal_feats + feats_to_merge]
    for feat, value in errors.items():
        logging.error(f'feature {feat} has {len(value)} errors.')
    if keep:
        return datum
    else:
        return list()


def merge(df, progress=False):

    data = list()
    errors = defaultdict(list)
    iterator = df.iterrows()
    if progress:
        iterator = tqdm(iterator)
    for r, s in iterator:
        ipa = s['ipa']
        datum = merge_ipa(s, ipa, s['segment'])
        if datum:
            data.append(datum)

    merged_df = pd.DataFrame(data, columns=['segment', 'ipa', 'merged_ipa'] + normal_feats + feats_to_merge)
    merged_df['merged_ipa'] = merged_df['merged_ipa'].apply(lambda l: [''.join([str(lll) for lll in ll]) for ll in l])
    merged_df['ipa'] = merged_df['ipa'].apply(lambda l: [str(ll) for ll in l])
    merged_df['ipa_segment'] = _apply(merged_df['merged_ipa'], lambda lst: '-'.join(lst), progress=progress)
    return merged_df


def indexify_ipa(col: str, lst: List) -> List:
    cat_cls = Category.get_enum(col)
    return [getattr(cat_cls, x.replace('-', '_').upper()).value.g_idx for x in lst]


def indexify(df, progress=False):
    for feat in Category:
        col = feat.name.lower()
        new_col = f'{col}_idx'
        df[new_col] = _apply(df[col], lambda lst, col=col: indexify_ipa(col, lst), progress=progress)


def get_feat_matrix(s: Union[pd.Series, Segment]) -> torch.LongTensor:
    arr = np.stack([s[col] for col in idx_col_names], axis=1)
    tensor = torch.from_numpy(arr)
    return tensor


idx_col_names = [f'{feat.name.lower()}_idx' for feat in Category]


def get_pth_content(df, progress=False):
    filtered = df[['ipa_segment', 'merged_ipa'] + idx_col_names]

    segments = filtered['ipa_segment'].values
    matrices = list()
    iterator = filtered.iterrows()
    if progress:
        iterator = tqdm(iterator, total=len(filtered))
    for r, s in iterator:
        tensor = get_feat_matrix(s)
        matrices.append(tensor)
    out = {
        'segments': segments,
        'matrices': matrices
    }
    return out


def pipeline(source: Source, progress=False):
    cnt, _, df = get_ipa_data(source, progress=progress)
    if cnt > 0:
        raise RuntimeError(f'Some tokens are invalid.')

    apply_all(df, progress=progress)
    cleaned_df = clean_data(df, progress=progress)
    merged_df = merge(cleaned_df, progress=progress)
    indexify(merged_df, progress=progress)
    out = get_pth_content(merged_df, progress=progress)
    return out
