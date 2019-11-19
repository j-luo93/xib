from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from typing import Callable, Iterator, List, Sequence, TextIO, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ipapy.ipachar import (DG_C_MANNER, DG_C_PLACE, DG_C_VOICING,
                           DG_DIACRITICS, DG_S_BREAK, DG_S_LENGTH, DG_S_STRESS,
                           DG_T_CONTOUR, DG_T_GLOBAL, DG_T_LEVEL, DG_TYPES,
                           DG_V_BACKNESS, DG_V_HEIGHT, DG_V_ROUNDNESS)
from ipapy.ipastring import IPAString
from tqdm import tqdm

from xib.ipa import Category

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


class Segment:

    def __init__(self, token: str):
        self.ipa = get_string(token)
        self.token = token
        self._merged = False
        if len(self.ipa) == 0:
            raise ValueError('Invalid IPA string.')
        self._apply_all()
        self._merge()
        self._indexify()
        self.feat_matrix = self._get_feat_matrix()

    def _apply_all(self):
        for name, dg in name2dg.items():
            setattr(self, name, get_dg_value(self.ipa, dg))
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
        datum = merge_ipa(self, self.ipa, self.token)
        if not datum:
            raise ValueError('Invalid IPA string.')
        self.merged_ipa = datum[2]
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

    def _get_feat_matrix(self) -> torch.LongTensor:
        return get_feat_matrix(self)


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
        datum = merge_ipa(s, ipa)
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
        df[new_col] = _apply(df[col], lambda col: indexify_ipa(col, lst), progress=progress)


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
