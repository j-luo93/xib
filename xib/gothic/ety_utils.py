from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import (Dict, Iterable, List, NewType, Optional, Sequence, Set,
                    Tuple)

import pandas as pd

from xib.gothic.core import Lang, Token, get_token, process_table

IDG_PROPAGATE = False

Word = NewType('Word', str)
RelType = NewType('RelType', str)
LangSeq = Sequence[Lang]
WordSeq = Sequence[Word]
RelTypeSeq = Sequence[RelType]


class EtymologicalDictionary:
    """Represents a dictionary that stores etymological information.

    This follows a list of conventions:
    1. The information is represented by a 5-tuple:
            (lang1, word1, lang2, word2, rel_type)
       This means that word1 in lang1 is "connected" with word2 in lang2 where a connection might
       mean borrowing or cognation, indicated by the value of `rel_type`. Note that it is not directional.
    2. These tuples are stored in a DataFrame instance. Column names are "lang1", "word1", "lang2", "word2" and "rel_type".
    """

    def __init__(self, name: str, lang1_seq: LangSeq, word1_seq: WordSeq, lang2_seq: LangSeq, word2_seq: WordSeq, rel_type_seq: RelTypeSeq):
        self.name = name
        self.data = pd.DataFrame({
            'lang1': lang1_seq,
            'word1': word1_seq,
            'lang2': lang2_seq,
            'word2': word2_seq,
            'rel_type': rel_type_seq
        })
        # Drop duplicates.
        self.data = self.data.drop_duplicates().reset_index(drop=True)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str, lang1: str, lang2: str) -> EtymologicalDictionary:
        """Convert a normal data frame into an EtymologicalDictionoary instance.

        The input `df` must have columns corresponding to the languages in question, and also a column `rel_type`.
        """
        word1_seq = [get_token(w, lang1) for w in df[lang1]]
        word2_seq = [get_token(w, lang2) for w in df[lang2]]
        lang1_seq = [lang1] * len(word1_seq)
        lang2_seq = [lang2] * len(word2_seq)
        rel_type_seq = df['rel_type']
        ety_dict = EtymologicalDictionary(name, lang1_seq, word1_seq, lang2_seq, word2_seq, rel_type_seq)
        return ety_dict

    def count_langs(self):
        print(self.data['lang1'].value_counts())
        print('-' * 30)
        print(self.data['lang2'].value_counts())
        print('-' * 30)


class _CognateSet:
    """A set of cognates."""

    def __init__(self):
        self.tokens: Set[Token] = set()
        self._lang2tokens: Dict[Lang, Set[Token]] = defaultdict(set)

    def __repr__(self):
        return f'CognateSet: size {len(self.tokens)}'

    def add(self, token: Token):
        self.tokens.add(token)
        self._lang2tokens[token.lang].add(token)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        yield from self.tokens

    def __contains__(self, token: Token):
        return token in self.tokens

    def has_lang(self, lang: Lang) -> bool:
        """Return whether any token in `lang` is present in this cognate set."""
        return lang in self._lang2tokens


@dataclass
class Coverage:
    num_covered: int
    total: int
    covered: List[Token] = field(repr=False)
    not_covered: List[Token] = field(repr=False)
    ratio: float = field(init=False)

    def __post_init__(self):
        self.ratio = float(f'{(self.num_covered / self.total):.3f}')


_Edges = Dict[Token, Set[str]]


class TokenNotFound(Exception):
    """Raise this error if a token is not found in the graph."""


class EtymologicalGraph:
    """A graph representing the etymological relationship."""

    def __init__(self):
        self._tokens: Set[Token] = set()
        self._edges: Dict[Token, _Edges] = defaultdict(lambda: defaultdict(set))
        self._finalized = False
        self._token2cog_set: Dict[Token, _CognateSet] = dict()

    def add_cognate_pair(self, token1: Token, token2: Token, source: str):
        if self._finalized:
            raise RuntimeError(f'The graph has been finalized.')
        self._tokens.add(token1)
        self._tokens.add(token2)
        self._edges[token1][token2].add(source)
        self._edges[token2][token1].add(source)

    def add_etymological_dictionary(self, ety_dict: EtymologicalDictionary):
        for _, row in ety_dict.data.iterrows():
            self.add_cognate_pair(row['word1'], row['word2'], ety_dict.name)

    def _check_exists(self, token: Token):
        if token not in self._edges:
            raise TokenNotFound(f'{token!r} not found in the graph.')

    def __getitem__(self, token: Token) -> _Edges:
        self._check_exists(token)
        return self._edges[token]

    def get_cognate_set(self, token: Token, order: Optional[int] = None) -> _CognateSet:
        """If `order` is specified, only expand the cognate set by this number, and no caching will be used.

        For instance, `order == 1` means one edge away.
        """
        if order is None:
            cog_set = self._get_cognate_set(token)
            if self._finalized:
                for cog in cog_set:
                    self._token2cog_set[cog] = cog_set
        else:
            cog_set = self._get_cognate_set(token, order=order)
        return cog_set

    def _get_cognate_set(self, token: Token, order: Optional[int] = None) -> _CognateSet:
        self._check_exists(token)
        cog_set = _CognateSet()
        cog_set.add(token)
        queue: List[Tuple[Token, int]] = [(token, 0)]
        # NOTE(j_luo) Only propagate idg lemmas if this token is idg or the flag is set to True.
        idg_propagate = IDG_PROPAGATE or token.lang == 'idg'
        while queue:
            to_expand, dist = queue.pop(0)
            if to_expand.lang != 'idg' or idg_propagate:
                for neighbor in self[to_expand]:
                    if not neighbor in cog_set:
                        if order is None or dist < order:
                            queue.append((neighbor, dist + 1))
                            cog_set.add(neighbor)
        return cog_set

    def has_cognate_in(self, token: Token, lang: Lang) -> bool:
        """Return whether `token` has a cognate in `lang`."""
        cog_set = self.get_cognate_set(token)
        return cog_set.has_lang(lang)

    def compute_coverage(self, tokens: Iterable[Token], lang: Lang) -> Coverage:
        total = len(tokens)
        covered = list()
        not_covered = list()
        for token in tokens:
            try:
                _is_covered = self.has_cognate_in(token, lang)
                covered.append(token)
            except TokenNotFound:
                not_covered.append(token)
        num_covered = len(covered)
        return Coverage(num_covered, total, covered, not_covered)

    def get_all_cognate_sets(self, tokens: Iterable[Token]) -> pd.DataFrame:
        remaining = set(tokens)
        data = list()
        while remaining:
            token = remaining.pop()
            try:
                cog_set = self.get_cognate_set(token)
                for cog in cog_set:
                    if cog in remaining:
                        remaining.remove(cog)
                data.append((token, cog_set))
            except TokenNotFound:
                pass
        ret = pd.DataFrame(data, columns=['seed', 'cog_set'])
        return ret

    def finalize(self):
        self._finalized = True

    def definalize(self):
        self._finalized = False
        self._token2cog_set = dict()


def get_ety_dict(table, column: str, name: str):
    """Get an EtymologicalDictionary instance."""
    token_col = f'{column}_tokens'
    df = table.reset_index(drop=True).explode(token_col)
    df = df.reset_index(drop=True).explode('lang_codes')[['Lemma', 'Sprachen', 'lang_codes', token_col]]
    df = df.dropna()

    lang1_seq = df['Lemma']
    word1_seq = df['Sprachen']
    lang2_seq = df[token_col]
    word2_seq = df['lang_codes']
    rel_type = ['cog'] * len(lang1_seq)

    ety_dict = EtymologicalDictionary(name, word1_seq, lang1_seq, word2_seq, lang2_seq, rel_type)
    return ety_dict


def get_ety_dict_from_tsv(tsv_path: str, column: str, name: str):
    table = process_table(tsv_path, column)
    ety_dict = get_ety_dict(table, column, name)
    return ety_dict
