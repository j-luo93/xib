import re
import sys

import pandas as pd

from xib.gomorph import RE_C

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


def get_germ_stem(s):
    ret = s
    while True:
        # Keep stripping away endings until nothing can be stripped.
        stripped = False
        for ending in germ_to_remove:
            if ret.endswith(ending) and len(ending) <= 0.5 * len(ret):
                if ending not in germ_to_match or re.search(germ_to_match[ending], ret):
                    ret = ret.rstrip(ending)
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
                        ret = ret.rstrip(ending)
                        stripped = True
            # Double ending with 'bb' should be mapped to just 'b'.
            if len(ret) >= 4 and ret[-1] == ret[-2] == 'b':
                stripped = True
                ret = ret[:-1]
            if not stripped:
                break
    return f'0~{len(ret) - 1}@{ret}:{s}'


if __name__ == "__main__":
    in_path, out_path, lang = sys.argv[1:]

    with open(in_path, 'r', encoding='utf8') as fin:
        df = pd.DataFrame({'Token': [line.strip() for line in fin]})

    lang2func = {
        'germ': get_germ_stem,
        'ang': get_ang_stem
    }
    df['Stems'] = df['Token'].apply(lang2func[lang])

    df.to_csv(out_path, index=None, sep='\t')