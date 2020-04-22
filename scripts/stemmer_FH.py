import re
import sys

import pandas as pd

RE_HV = 'ƕ'
RE_CHAR = fr'[\wïþÞ{RE_HV}]'
RE_C = fr'[bcdfghjklmnpqrstwxzþðÞ{RE_HV}ċġ]'
RE_Di = fr'[a]'

# ------------------- Proto-Germainic helper ------------------- #

# Endings to remove.
germ_to_remove = {
    'az', 'ą', 'ō', 'ī', 'jō', 'iz', 'i', 'uz', 'u', 'ô', 'ǭ', 'īn', 'ēr', 'jaz', 'ja', 'jan', 'wan', 'an'
}
germ_to_remove = sorted(germ_to_remove, key=len, reverse=True)

# For some endings, you need to meet certain conditions before applying the mapping. For instance, strip the ending 'u' only if it's preceded by some consonant.
germ_to_match = {
    'u': fr'(?<={RE_C})u$',
    'uz': fr'(?<={RE_C})uz$',
}
germ_to_prevent = {
    'u': fr'(?<={RE_Di})u$',
    'uz': fr'(?<={RE_Di})uz$',
    'i': fr'(?<={RE_Di})i$',
    'iz': fr'(?<={RE_Di})iz$',
}


def get_germ_stem(s):
    ret = s
    while True:
        # Keep stripping away endings until nothing can be stripped.
        stripped = False
        for ending in germ_to_remove:
            if ret.endswith(ending) and len(ending) <= 0.5 * len(ret):
                if ending not in germ_to_match or re.search(germ_to_match[ending], ret):
                    if ending not in germ_to_prevent or not re.search(germ_to_prevent[ending], ret):
                        #ret = ret.rstrip(ending)
                        ret = re.sub(ending + '$', "", ret)
                        stripped = True
        if not stripped:
            break
    return f'0~{len(ret) - 1}@{ret}:{s}'

# --------------------- Old English helper --------------------- #


ang_to_remove = {
    'an', 'u', 'a', 'e', 'h', 'ian'
}
ang_to_remove = sorted(ang_to_remove, key=len, reverse=True)

ang_to_match = {
    'an': fr'(?<={RE_C})an$',
    'u': fr'(?<={RE_C})u$',
    'a': fr'(?<={RE_C})a$',
    'e': fr'(?<={RE_C})e$',
    'ian': fr'(?<={RE_C})ian$',
}
ang_exceptions = {
    'oþþe': 'oþþe',
    'hēan': 'hēan'
}


def get_ang_stem(s):
    ret = s
    if s in ang_exceptions:
        ret = ang_exceptions[s]
    elif not s.startswith('-'):
        while True:
            stripped = False
            for ending in ang_to_remove:
                if ret.endswith(ending) and len(ending) <= 0.5 * len(ret) and len(ret) >= 4:
                    if ending not in ang_to_match or re.search(ang_to_match[ending], ret):
                        #ret = ret.rstrip(ending)
                        ret = re.sub(ending + '$', "", ret)
                        stripped = True

            # Double ending with 'bb' should be mapped to just 'b'.
            if len(ret) >= 4 and ret[-1] == ret[-2] == 'b':
                stripped = True
                ret = ret[:-1]
            if not stripped:
                break
    return f'0~{len(ret) - 1}@{ret}:{s}'

# --------------------- Old Norse helper --------------------- #


on_to_remove = {
    'a', 'ja', 'ir', 'i', 'ur', 'ar', 'ast', 'r'
}
on_to_remove = sorted(on_to_remove, key=len, reverse=True)

on_to_match = {
    'r': fr'(?<={RE_C})r$',
}
on_exceptions = {
    # 'oþþe': 'oþþe',
    # 'hēan': 'hēan'
    'aptr': 'aptr'
}


def get_on_stem(s):
    if '=' in s or 'ᛏ' in s:
        return None
    ret = s
    if s in on_exceptions:
        ret = on_exceptions[s]
    elif not s.startswith('-'):
        # Maybe just strip once.
        # while True:
        stripped = False
        for ending in on_to_remove:
            if ret.endswith(ending) and len(ending) <= 0.5 * len(ret) and len(ret) >= 2:
                if ending not in on_to_match or re.search(on_to_match[ending], ret):
                    #ret = ret.rstrip(ending)
                    ret = re.sub(ending + '$', "", ret)
                    stripped = True
                    break

        # Double ending with 'bb' should be mapped to just 'b'.
        if len(ret) >= 4 and ret[-1] == ret[-2] == 'b':
            stripped = True
            ret = ret[:-1]
        # If not stripped already, change double n's or l's to just single n or l.
        if not stripped and (ret.endswith('nn') or ret.endswith('ll')):
            ret = ret[:-1]

        # ret = re.sub('ey', "ai", ret)
        # ret = re.sub('ju', "eu", ret)
        # ret = re.sub('jo', "eu", ret)
        # ret = re.sub('ja', "e", ret)
        # ret = re.sub('jǫ', "e", ret)
        # ret = re.sub('ø', "e", ret)
        # ret = re.sub('æ', "e", ret)
        # ret = re.sub('o', "u", ret)
        # ret = re.sub('ǫ', "a", ret)
        # ret = re.sub('y', "u", ret)
        # ret = re.sub('y', "u", ret)
    return f'0~{len(ret) - 1}@{ret}:{s}'


if __name__ == "__main__":
    in_path, out_path, lang = sys.argv[1:]

    with open(in_path, 'r', encoding='utf8') as fin:
        df = pd.DataFrame({'Token': [line.strip() for line in fin]})

    lang2func = {
        'germ': get_germ_stem,
        'ang': get_ang_stem,
        'on': get_on_stem
    }
    df['Stems'] = df['Token'].apply(lang2func[lang])

    df = df.dropna(subset=['Stems'])

    df.to_csv(out_path, index=None, sep='\t')
