from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from typing import (Callable, ClassVar, Iterator, List, Optional, Sequence,
                    TextIO, Tuple, Union)

import numpy as np
import pandas as pd
import torch
from ipapy.ipachar import (DG_C_MANNER, DG_C_PLACE, DG_C_VOICING,
                           DG_DIACRITICS, DG_S_BREAK, DG_S_LENGTH, DG_S_STRESS,
                           DG_T_CONTOUR, DG_T_GLOBAL, DG_T_LEVEL, DG_TYPES,
                           DG_V_BACKNESS, DG_V_HEIGHT, DG_V_ROUNDNESS)
from ipapy.ipastring import IPAString
from tqdm import tqdm

from dev_misc import add_argument, g
from dev_misc.devlib import BT, LT
from dev_misc.utils import cached_property, deprecated
from xib.ipa import Category

B, I, O = 0, 1, 2

tqdm.pandas()


dia2char = {
    'low': {'à': 'a', 'è': 'e', 'ò': 'o', 'ì': 'i', 'ù': 'u', 'ѐ': 'e', 'ǹ': 'n', 'ỳ': 'y'},
    'high': {'á': 'a', 'é': 'e', 'ó': 'o', 'ú': 'u', 'ý': 'y', 'í': 'i', 'ḿ': 'm', 'ĺ': 'l',
             'ǿ': 'ø', 'ɔ́': 'ɔ', 'ɛ́': 'ɛ', 'ǽ': 'æ', 'ə́': 'ə', 'ŕ': 'r', 'ń': 'n'},
    'rising_falling': {'ã': 'a'},
    'falling': {'â': 'a', 'î': 'i', 'ê': 'e', 'û': 'u', 'ô': 'o', 'ŷ': 'y', 'ĵ': 'j'},
    'rising': {'ǎ': 'a', 'ǐ': 'i', 'ǔ': 'u', 'ǒ': 'o', 'ě': 'e'},
    'extra_short': {'ă': 'a', 'ĕ': 'e', 'ĭ': 'i', 'ŏ': 'o', 'ŭ': 'u'},
    'nasalized': {'ĩ': 'i', 'ũ': 'u', 'ã': 'a', 'õ': 'o', 'ẽ': 'e', 'ṽ': 'v', 'ỹ': 'y'},
    'breathy_voiced': {'ṳ': 'u'},
    'creaky_voiced': {'a̰': 'a', 'ḭ': 'i', 'ḛ': 'e', 'ṵ': 'u'},
    'centralized': {'ë': 'e', 'ä': 'a', 'ï': 'i', 'ö': 'o', 'ü': 'u', 'ÿ': 'y'},
    'mid': {'ǣ': 'æ', 'ū': 'u', 'ī': 'i', 'ē': 'e', 'ā': 'a', 'ō': 'o'},
    'voiceless': {'ḁ': 'a'},
    'extra_high': {'ő': 'o'},
    'extra_low': {'ȁ': 'a'},
    'syllabic': {'ạ': 'a', 'ụ': 'u'}
}

dia2code = {
    'low': 768,
    'high': 769,
    'rising_falling': 771,
    'falling': 770,
    'rising': 780,
    'extra_short': 774,
    'nasalized': 771,
    'breathy_voiced': 804,
    'creaky_voiced': 816,
    'centralized': 776,
    'mid': 772,
    'voiceless': 805,
    'extra_high': 779,
    'extra_low': 783,
    'syllabic': 809,
    'high_rising': 7620,
    'low_rising': 7621,
}

char2ipa_char = dict()
for dia, char_map in dia2char.items():
    code = dia2code[dia]
    s = chr(code)
    for one_char, vowel in char_map.items():
        char2ipa_char[one_char] = vowel + s


to_remove = {'ᶢ', '̍', '-', 'ⁿ', 'ᵑ', 'ᵐ', 'ᶬ', ',', 'ᵊ', 'ˢ', '~', '͍', 'ˣ', 'ᵝ', '⁓', '˭', 'ᵈ', '⁽', '⁾', '˔', 'ᵇ',
             '+', '⁻'}


def clean(s):
    if s == '◌̃':
        return ''
    return ''.join(c for c in s if c not in to_remove)


def sub(s):
    return ''.join(char2ipa_char.get(c, c) for c in s)


to_standardize = {
    'ˁ': 'ˤ',
    "'": 'ˈ',
    '?': 'ʔ',
    'ṭ': 'ʈ',
    'ḍ': 'ɖ',
    'ṇ': 'ɳ',
    'ṣ': 'ʂ',
    'ḷ': 'ɭ',
    ':': 'ː',
    'ˇ': '̌',
    'ỵ': 'y˞',
    'ọ': 'o˞',
    'ř': 'r̝',  # Czech
    '͈': 'ː',  # Irish
    'ŕ̩': sub('ŕ') + '̩',  # sanskrit
    'δ': 'd',  # Greek
    'ń̩': sub('ń') + '̩',  # unsure
    'ε': 'e',
    'X': 'x',
    'ṍ': sub('õ') + chr(769),
    'ÿ̀': sub('ÿ') + chr(768),
    '∅': 'ʏ'  # Norvegian,
}


def standardize(s):
    return ''.join(to_standardize.get(c, c) for c in s)


def get_string(s: str) -> IPAString:
    return IPAString(unicode_string=clean(sub(standardize(s))))


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
        if self.ptype[0] not in ['consonant', 'vowel']:
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


class SegmentWindow(BaseSegmentWithGoldTagSeq):

    def __init__(self, segments: List[Union[Segment, BrokenSegment]], original: Optional[SegmentWindow] = None):
        self._segments = segments
        self.original = original

    def __len__(self):
        return sum(len(segment) for segment in self._segments)

    @cached_property
    def merged_ipa(self) -> List[IPAString]:
        ret = list()
        for segment in self._segments:
            ret.extend(segment.merged_ipa)
        return ret

    @property
    def gold_tag_seq(self) -> LT:
        return torch.cat([segment.gold_tag_seq for segment in self._segments], dim=0)

    @cached_property
    def feat_matrix(self) -> LT:
        matrices = [segment.feat_matrix for segment in self._segments]
        return torch.cat(matrices, dim=0)

    def __str__(self):
        return ' '.join(str(segment) for segment in self._segments)

    def __getitem__(self, idx: int):
        if not isinstance(idx, (np.integer, int)):
            raise TypeError(f'Expecting int/np.integer, but got {type(idx)}')
        if idx >= len(self):
            raise IndexError(f'Index {idx} out of bound for length {len(self)}.')

        seg_idx, idx_in_seg = self._find_segment(idx)
        return self._segments[seg_idx][idx_in_seg]

    def _find_segment(self, idx: int) -> Tuple[int, int]:
        length = 0
        for seg_idx, segment in enumerate(self._segments):
            length += len(segment)
            if length > idx:
                break
        idx_in_seg = idx - (length - len(segment))
        return seg_idx, idx_in_seg

    def to_segmentation(self) -> Segmentation:
        spans = list()
        offset = 0
        for segment in self._segments:
            span = segment.to_span()
            if span is not None:
                span.start += offset
                span.end += offset
                spans.append(span)
            offset += len(segment)
        return Segmentation(spans)

    @cached_property
    def segment_list(self) -> List[str]:
        ret = list()
        for segment in self._segments:
            ret.extend(segment.segment_list)
        return ret

    def get_segmentation_from_tags(self, tags: Sequence[int]) -> Segmentation:
        assert len(tags) >= len(self)

        i = 0
        spans = list()
        while i < len(self):
            t_i = tags[i]
            if t_i == O:
                i += 1
            elif t_i == I or t_i == B:
                j = i + 1
                while j < len(self) and tags[j] == I:
                    j += 1
                value = self.segment_list[i: j]
                span = Span(value, i, j - 1)
                spans.append(span)
                i = j
        return Segmentation(spans)

    def perturb(self, mode) -> PerturbedSegment:
        assert mode in ['swap', 'shift']
        if mode == 'swap':
            return self.perturb_swap()
        else:
            return self.perturb_shift()

    def perturb_swap(self) -> PerturbedSegment:
        if len(self) <= 1:
            return self
        # Swap two consecutive units.
        pos = random.randint(0, len(self) - 2)
        left = self.segment_list[:pos]
        mid = [self.segment_list[pos + 1], self.segment_list[pos]]
        right = self.segment_list[pos + 2:]
        new_list_of_units = left + mid + right

        left_ipa = self.merged_ipa[:pos]
        mid_ipa = [self.merged_ipa[pos + 1], self.merged_ipa[pos]]
        right_ipa = self.merged_ipa[pos + 2:]
        new_list_of_ipas = left_ipa + mid_ipa + right_ipa

        left_m = self.feat_matrix[:pos]
        mid_m = [self.feat_matrix[pos + 1: pos + 2], self.feat_matrix[pos: pos + 1]]
        right_m = self.feat_matrix[pos + 2:]
        new_feat_matrix = torch.cat([left_m] + mid_m + [right_m], dim=0)
        return PerturbedSegment(new_list_of_units, new_list_of_ipas, new_feat_matrix)

    def perturb_shift(self) -> PerturbedSegment:
        if len(self) <= 1:
            return self
        shift = random.randint(1, len(self) - 1)
        mid_pt = len(self) - shift
        left = self.segment_list[mid_pt:]
        right = self.segment_list[:mid_pt]
        new_list_of_units = left + right

        left_ipa = self.merged_ipa[mid_pt:]
        right_ipa = self.merged_ipa[:mid_pt]
        new_list_of_ipas = left_ipa + right_ipa

        left_m = self.feat_matrix[mid_pt:]
        right_m = self.feat_matrix[:mid_pt]
        new_feat_matrix = torch.cat([left_m, right_m], dim=0)
        return PerturbedSegment(new_list_of_units, new_list_of_ipas, new_feat_matrix)

    def perturb_n_times(self, times: int) -> Tuple[List[PerturbedSegment], List[bool]]:
        """This will return a list of perturbed segments, as well as the original segment."""
        segment_set = set()
        duplicated = list()
        segments = list()

        def update(seg):
            if seg not in segment_set:
                duplicated.append(False)
                segment_set.add(seg)
            else:
                duplicated.append(True)
            segments.append(seg)

        update(self)

        for _ in range(times):
            ptb_segment_swap = self.perturb_swap()
            update(ptb_segment_swap)
            ptb_segment_shift = self.perturb_shift()
            update(ptb_segment_shift)

        return segments, duplicated

    def break_segment(self, start: int, end: int) -> SegmentWindow:
        if start == 0 and end == len(self) - 1:
            return self

        start_seg_idx, idx_in_start = self._find_segment(start)
        end_seg_idx, idx_in_end = self._find_segment(end)

        if start_seg_idx == end_seg_idx:
            seg = self._segments[start_seg_idx]
            broken = seg.break_segment(idx_in_start, idx_in_end)
            original = SegmentWindow([seg])
            return SegmentWindow([broken], original=original)
        else:
            start_seg = self._segments[start_seg_idx]
            end_seg = self._segments[end_seg_idx]
            broken_start = start_seg.break_segment(idx_in_start, len(start_seg) - 1)
            broken_end = end_seg.break_segment(0, idx_in_end)
            middle = [self._segments[i] for i in range(start_seg_idx + 1, end_seg_idx)]
            original = SegmentWindow([start_seg] + middle + [end_seg])
            return SegmentWindow([broken_start] + middle + [broken_end], original=original)


class BaseSpecialSegment(BaseSegment):

    def __init__(self, list_of_units: List[str], list_of_ipas: List[IPAString], feat_matrix: LT):
        self._list_of_units = list_of_units
        self._list_of_ipas = list_of_ipas
        self._feat_matrix = feat_matrix
        if len(self._list_of_units) != len(self._feat_matrix):
            raise ValueError(f'Length mismatch.')

    @property
    def merged_ipa(self) -> List[IPAString]:
        return self._list_of_ipas

    @property
    def feat_matrix(self):
        return self._feat_matrix

    def __len__(self):
        return len(self._list_of_units)

    def __getitem__(self, idx: int) -> str:
        return self._list_of_units[idx]

    @property
    def segment_list(self):
        return self._list_of_units


class PerturbedSegment(BaseSpecialSegment):

    has_gold_tag_seq: ClassVar[bool] = False

    def __str__(self):
        return '!' + '-'.join(self._list_of_units)


class BrokenSegment(BaseSpecialSegment, BaseSegmentWithGoldTagSeq):

    def __init__(self, list_of_units: List[str], list_of_ipas: List[IPAString], feat_matrix: LT, gold_tag_seq: LT, original: Segment):
        super().__init__(list_of_units, list_of_ipas, feat_matrix)
        self._gold_tag_seq = gold_tag_seq
        if len(self._gold_tag_seq) != len(list_of_units):
            raise ValueError('Length mismatch.')
        self.original = original

    @property
    def gold_tag_seq(self) -> LT:
        return self._gold_tag_seq

    def __str__(self):
        return ']' + '-'.join(self._list_of_units) + '['

    def to_span(self) -> None:
        return


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
                        errors[(feat)].append(s)
                        keep = False
            j += 1
        merged_ipa.append(ipa[i:j])
        i = j
        for feat in feats_to_merge:
            datum_cols[feat].append(de_none(datum_c_to_merge[feat]))
    datum = [segment, ipa, merged_ipa] + [datum_cols[feat] for feat in normal_feats + feats_to_merge]
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
