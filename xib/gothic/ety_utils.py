from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heapify, heappop
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
       mean borrowing or cognation, indicated by the value of `rel_type`. Note that it is directional.
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

    def __init__(self, root: Token):
        self.root: Token = root
        self.tokens: Set[Token] = set()
        self._lang2tokens: Dict[Lang, Set[Token]] = defaultdict(set)
        self.add(self.root)

    def __repr__(self):
        return f'CognateSet(size={len(self.tokens)})'

    def add(self, token: Token):
        self.tokens.add(token)
        self._lang2tokens[token.lang].add(token)

    def set_root(self, root: Token):
        self.add(root)
        self.root = root

    def merge_with_root_cog_set(self, root_cog_set: _CognateSet):
        for token in self:
            root_cog_set.add(token)

    def copy(self) -> _CognateSet:
        obj = _CognateSet(self.root)
        for token in self.tokens:
            obj.add(token)
        return obj

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
        self.tokens: Set[Token] = set()  # Stores all tokens
        self.out_edges: Dict[Token, _Edges] = defaultdict(lambda: defaultdict(set))  # Stores all out-going edges
        self.in_edges: Dict[Token, _Edges] = defaultdict(lambda: defaultdict(set))  # Stores all in-going edges.
        # Whether this graph is finalized. Finalizing the graph will initiate the process of building cognate sets.
        self._finalized = False
        # Stores the mapping from tokens to cognate sets. Note that one token can belong to several cognate sets and this is only filled in after finalization.
        self._token2cog_sets: Dict[Token, List[_CognateSet]] = defaultdict(list)

    def _check_finalized(self, value: bool):
        if self._finalized != value:
            if value:
                raise RuntimeError(f'Not finalized.')
            else:
                raise RuntimeError(f'Already finalized.')

    def finalize(self):
        self._check_finalized(False)
        # Propapate all tokens, in ascending order of out degree of the nodes.
        out_degrees = {token: len(self.out_edges[token]) for token in self.tokens}
        queue = [token for token, od in out_degrees.items() if od == 0]
        # Stores the mapping only for recently propagated tokens.
        token2cog_set = {token: _CognateSet(token) for token in queue}
        visited = set()
        while queue:
            v = queue.pop(0)
            visited.add(v)
            cog_set = token2cog_set[v]
            in_degree = len(self.in_edges[v])
            cog_sets = [cog_set] + [cog_set.copy() for _ in range(in_degree - 1)]
            # NOTE(j_luo) Do not propagate to idg nodes if not configured to do so.
            # NOTE(j_luo) ga is just weird. Many entries are wrong.
            in_edges = {u: sources for u, sources in self.in_edges[v].items() if (
                IDG_PROPAGATE or u.lang != 'idg') and not u.is_same_string('ga')}
            if in_edges:
                # Remove the copy for v if v is not some root, which means it will be propagated through. Only the roots should have a copy left after all this proces.
                del token2cog_set[v]
                for cog_set, u in zip(cog_sets, in_edges):
                    if u in token2cog_set:
                        # If u is already propagated to by another child, merge with it.
                        root_cog_set = token2cog_set[u]
                        cog_set.merge_with_root_cog_set(root_cog_set)
                    else:
                        # Set u as the new root.
                        cog_set.set_root(u)
                        # Set cog_set as u's new cognate set.
                        token2cog_set[u] = cog_set
                    out_degrees[u] -= 1
                    if out_degrees[u] == 0:
                        queue.append(u)
        diff = len(set(self.tokens) - set(visited))
        if diff:
            logging.warning(
                f'{diff} nodes not visited. Either they are part of a cyclic graph, or isolated due to removed self loops, or not permitted to propagate through idg nodes.')
        # Compute self._token2cog_sets.
        for root, cog_set in token2cog_set.items():
            for token in cog_set:
                self._token2cog_sets[token].append(cog_set)

        self._finalized = True

    def definalize(self):
        self._finalized = False
        self._token2cog_sets = dict()

    def add_cognate_pair(self, token1: Token, token2: Token, source: str):
        if self._finalized:
            raise RuntimeError(f'The graph has been finalized.')
        if token1 != token2:  # NOTE(j_luo) Avoid self loops.
            self.tokens.add(token1)
            self.tokens.add(token2)
            self.out_edges[token1][token2].add(source)
            self.in_edges[token2][token1].add(source)

    def add_etymological_dictionary(self, ety_dict: EtymologicalDictionary):
        for _, row in ety_dict.data.iterrows():
            self.add_cognate_pair(row['word1'], row['word2'], ety_dict.name)

    def get_cognate_sets(self, token: Token) -> List[_CognateSet]:
        self._check_finalized(True)
        self._check_exists(token)
        return self._token2cog_sets[token]

    def _check_exists(self, token: Token):
        if token not in self.tokens:
            raise TokenNotFound(f'{token!r} not found in the graph.')

    def __getitem__(self, token: Token) -> Tuple[_Edges, _Edges]:
        self._check_exists(token)
        return (self.out_edges[token], self.in_edges[token])

    def has_cognate_in(self, token: Token, lang: Lang) -> bool:
        """Return whether `token` has a cognate in `lang`."""
        cog_sets = self.get_cognate_sets(token)
        return any(cog_set.has_lang(lang) for cog_set in cog_sets)

    def compute_coverage(self, tokens: Iterable[Token], lang: Lang) -> Coverage:
        total = len(tokens)
        covered = list()
        not_covered = list()
        for token in tokens:
            try:
                is_covered = self.has_cognate_in(token, lang)
            except TokenNotFound:
                is_covered = False
            finally:
                if is_covered:
                    covered.append(token)
                else:
                    not_covered.append(token)
        num_covered = len(covered)
        return Coverage(num_covered, total, covered, not_covered)


def get_ety_dict(table, column: str, name: str):
    """Get an EtymologicalDictionary instance."""
    token_col = f'{column}_tokens'
    # Default to the lemma language if lang_codes is not present.
    df = table[['Lemma', 'Sprachen', 'lang_codes', 'is_ref', 'is_single_ref', token_col]]
    df['lang_codes'] = df[['Sprachen', 'lang_codes']].apply(lambda item: item[1] if item[1] else [item[0]], axis=1)
    df = df.reset_index(drop=True).explode(token_col)
    df = df.reset_index(drop=True).explode('lang_codes')
    df = df.dropna()
    df = df[(~df['is_ref']) | (df['is_single_ref'])]
    df[token_col] = df[['is_single_ref', token_col]].apply(
        lambda item: item[1].tokens[0] if item[0] else item[1], axis=1)

    lang1_seq = df['Lemma']
    word1_seq = df['Sprachen']
    lang2_seq = df[token_col]
    word2_seq = df['lang_codes']
    rel_type = df['is_ref'].apply(lambda is_ref: 'ref' if is_ref else 'cog')
    if column == 'E':
        lang1_seq, lang2_seq = lang2_seq, lang1_seq
        word1_seq, word2_seq = word2_seq, word1_seq

    ety_dict = EtymologicalDictionary(name, word1_seq, lang1_seq, word2_seq, lang2_seq, rel_type)
    return ety_dict


def get_ety_dict_from_tsv(tsv_path: str, column: str, name: str):
    table = process_table(tsv_path, column)
    ety_dict = get_ety_dict(table, column, name)
    return ety_dict
