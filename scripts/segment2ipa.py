import sys
from collections import Counter, defaultdict
from pathlib import Path

import inflection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
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


def get_string(s):
    return IPAString(unicode_string=clean(sub(standardize(s))))


def apply(df, dg, col_name):
    df[col_name] = df['ipa'].progress_apply(lambda s: [c.dg_value(dg) for c in s.ipa_chars])


def de_none(s):
    return 'none' if s is None else s


if __name__ == "__main__":
    in_path = Path(sys.argv[1])
    lang = sys.argv[2]
    out_path = Path(sys.argv[3])
    assert lang, 'Specify lang'

    with in_path.open('r', encoding='utf8') as fin:
        cnt = 0
        total = 0
        data = list()
        for line in tqdm(fin):
            token = line.strip()
            try:
                ipa = get_string(token)
                data.append((token, ipa))
            except ValueError:
                cnt += 1
            total += 1
            if total == 1000:
                break

    print(f'Ignore {cnt} / {total} lines.')

    df = pd.DataFrame(data, columns=['segment', 'ipa'])

    # ptype stands for phone type.
    apply(df, DG_TYPES, 'ptype')
    # Consonants have three features: voicing, place, and manner
    apply(df, DG_C_VOICING, 'c_voicing')
    apply(df, DG_C_PLACE, 'c_place')
    apply(df, DG_C_MANNER, 'c_manner')
    # Vowels have three features: height, backness, and roundness.
    apply(df, DG_V_HEIGHT, 'v_height')
    apply(df, DG_V_BACKNESS, 'v_backness')
    apply(df, DG_V_ROUNDNESS, 'v_roundness')
    # Diacritics are grouped into one.
    apply(df, DG_DIACRITICS, 'diacritics')
    # Suprasegmentals have three groups: stress, length and break.
    apply(df, DG_S_STRESS, 's_stress')
    apply(df, DG_S_LENGTH, 's_length')
    apply(df, DG_S_BREAK, 's_break')
    # Tones have three groups:  level, contour and global.
    apply(df, DG_T_LEVEL, 't_level')
    apply(df, DG_T_CONTOUR, 't_contour')
    apply(df, DG_T_GLOBAL, 't_global')

    len_mask = (df['ipa'].str.len() > 0)
    clean_df = df[len_mask]

    # Some segments do not start with consonants or vowels.
    mask = clean_df['ptype'].progress_apply(lambda l: l[0] in ['consonant', 'vowel'])
    clean_df = clean_df[mask]

    clean_df.to_csv(f'phones_{lang}.tsv', sep='\t', index=False)

    normal_feats = ['ptype', 'c_voicing', 'c_place', 'c_manner', 'v_height', 'v_backness', 'v_roundness']
    feats_to_merge = ['diacritics', 's_stress', 's_length', 's_break', 't_level', 't_contour', 't_global']

    data = list()
    errors = defaultdict(list)
    for r, s in tqdm(clean_df.iterrows()):
        i = 0
        ptypes = s['ptype']
        ipa = s['ipa']
        segment = s['segment']
        keep = True
        datum_cols = {feat: list() for feat in normal_feats + feats_to_merge}
        merged_ipa = list()
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
            data.append(datum)

    merged_df = pd.DataFrame(data, columns=['segment', 'ipa', 'merged_ipa'] + normal_feats + feats_to_merge)
    merged_df['merged_ipa'] = merged_df['merged_ipa'].apply(lambda l: [''.join([str(lll) for lll in ll]) for ll in l])
    merged_df['ipa'] = merged_df['ipa'].apply(lambda l: [str(ll) for ll in l])

    # Save intermediate merged results.
    merged_df.to_csv(f'phones_merged_{lang}.tsv', sep='\t', index=False)

    merged_df['ipa_segment'] = merged_df['merged_ipa'].progress_apply(lambda lst: '-'.join(lst))

    for feat in Category:
        col = feat.name.lower()
        new_col = f'{col}_idx'
        cat_cls = Category.get_enum(col)
        merged_df[new_col] = merged_df[col].progress_apply(lambda lst: [getattr(
            cat_cls, x.replace('-', '_').upper()).value.g_idx for x in lst])

    col_names = [f'{feat.name.lower()}_idx' for feat in Category]

    filtered = merged_df[['ipa_segment', 'merged_ipa'] + col_names]

    segments = filtered['ipa_segment'].values
    matrices = list()
    for r, s in tqdm(filtered.iterrows(), total=len(filtered)):
        arr = np.stack([s[col] for col in col_names], axis=1)
        tensor = torch.from_numpy(arr)
        matrices.append(tensor)
    out = {
        'segments': segments,
        'matrices': matrices
    }
    torch.save(out, out_path)
