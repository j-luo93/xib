import re
from itertools import chain
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from lxml.html import HTMLParser, fromstring
from pyquery import PyQuery as pq

reserved_seps = dict()
group2col = dict()


def register_dataset_seps(dataset: str, seps: List[str]):
    reserved_seps[dataset] = seps
    group2col[dataset] = ['Lemma', 'Alternative', 'Sprachen', 'Wortarten_all', 'Bedeutungen_all'] + seps


def nm(dataset: str, sep: Optional[str] = None) -> str:  # Stands for negative match.
    to_avoid = reserved_seps[dataset]
    if sep is not None:
        to_avoid = reserved_seps[dataset][reserved_seps[dataset].index(sep) + 1:]
    reserved_p = ''.join([fr'(?!{sep}\.:)' for sep in to_avoid])
    return reserved_p


def get_regex(dataset: str, sep: Optional[str] = None) -> str:
    reserved_p = nm(dataset, sep)
    if sep is None:
        prefix = ''
    else:
        prefix = fr'{sep}\.:'
    return fr'({prefix}(?:{reserved_p}.)*;?\s*)?'


def get_match_regex(dataset: str) -> re.Pattern:
    lemma_p = r'^([^,]+),?\s*'
    alternative_p = r'((?:\W?[^\.]+\b)(?!\.),?\s*)*'  # r'((?:\b[^\.]+\b)(?!\.),?\s*)*'
    lang_p = r'([^\.]+\.,?\s*)'
    wortarten_p = r'([^:]+:\s*)?'
    bedeutung_p = get_regex(dataset)

    patterns = [lemma_p, alternative_p, lang_p, wortarten_p, bedeutung_p]
    for sep in reserved_seps[dataset][:-1]:
        patterns.append(get_regex(dataset, sep))
    # The last pattern needs special treatment.
    last_sep = reserved_seps[dataset][-1]
    patterns.append(fr'({last_sep}\.:(?:{nm(dataset, last_sep)}.)*)?\s*$')
    regex = re.compile(''.join(patterns))
    return regex


def clean_sep(prefix: str, s: str) -> str:
    """Clean notes/separators."""
    if s is not None:
        # Remove separators. Note that W.: might occur in the middle of the segment.
        s = re.sub(fr'{prefix}\.:', '', s)
        # Remove trailing ;
        s = re.sub(fr';$', '', s)
        s = s.strip()
    return s


def clean_lang(s: str) -> str:
    """Clean the language/Sprachen column."""
    return re.sub(r'[\s,\.]', '', s)


def merge_alternatives(item: Tuple[str, Optional[str]]) -> str:
    """Merge the main lemma with potential alternatives."""
    lemma, alt = item
    if alt is not None:
        lemma = lemma + ', ' + alt
    return lemma


def get_df_from_doc(doc: Iterable[str], dataset: str) -> pd.DataFrame:
    regex = get_match_regex(dataset)
    matches = list()
    for line in doc:
        matches.append(regex.match(line.strip()))

    # Extract columns from matches.
    data = list()
    for _, match in enumerate(matches):
        row = [match.group(idx) for idx, col in enumerate(group2col[dataset], 1)]
        data.append(tuple(row))

    df = pd.DataFrame(data, columns=group2col[dataset])
    df['Lemma'] = df[['Lemma', 'Alternative']].apply(merge_alternatives, axis=1)

    df['Sprachen'] = df['Sprachen'].apply(clean_lang)
    for sep in reserved_seps[dataset]:
        df[sep] = df[sep].apply(lambda s: clean_sep(sep, s))

    return df


def get_pq_doc(path: str, encoding: str):
    """PyQuery has some issue dealing with broken html files. See https://github.com/gawel/pyquery/issues/31."""
    parser = HTMLParser(encoding=encoding)

    with open(path, encoding=encoding) as fin:
        contents = fin.read()

    doc = pq(fromstring(contents, parser=parser))
    return doc


class LineIterable:

    def __init__(self, doc):
        self.doc = doc

    def __iter__(self):
        for elem in chain(self.doc('p.MsoNormal'), self.doc('p.MsoPlainText')):
            entry = pq(elem)
            text = entry.text()
            text = re.sub(r'\s', ' ', text).strip()  # EDEL has many multi-line contents.
            if text:  # Some entries are empty.
                yield text
