from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heapify, heappop
from typing import (Any, Dict, Iterable, List, NewType, Optional, Sequence,
                    Set, Tuple, Union)

import pandas as pd

from xib.gothic.core import Lang, MergedToken, Token, get_token, process_table

IDG_PROPAGATE = False


class RelType(Enum):
    SINGLE_REF = auto()
    COGNATE = auto()
    PREFIXED = auto()


@dataclass(eq=True, frozen=True)
class _BaseEdge:
    rel_type: RelType


@dataclass(eq=True, frozen=True)
class _BaseEdgeOptional:
    info: Optional[Any] = None


@dataclass(eq=True, frozen=True)
class _Edge(_BaseEdgeOptional, _BaseEdge):
    pass


@dataclass(eq=True, frozen=True)
class _BaseEdgeWithSource(_BaseEdge):
    source: str


@dataclass(eq=True, frozen=True)
class _EdgeWithSource(_BaseEdgeOptional, _BaseEdgeWithSource):
    pass


Word = NewType('Word', str)
LangSeq = Sequence[Lang]
WordSeq = Sequence[Word]
EdgeSeq = Sequence[_Edge]


class EtymologicalDictionary:
    """Represents a dictionary that stores etymological information.

    This follows a list of conventions:
    1. The information is represented by a 5-tuple:
            (lang1, word1, lang2, word2, rel_type)
       This means that word1 in lang1 is "connected" with word2 in lang2 where a connection might
       mean borrowing or cognation, indicated by the value of `rel_type`. Note that it is directional.
    2. These tuples are stored in a DataFrame instance. Column names are "lang1", "word1", "lang2", "word2" and "rel_type".
    """

    def __init__(self, name: str, lang1_seq: LangSeq, word1_seq: WordSeq, lang2_seq: LangSeq, word2_seq: WordSeq, edge_seq: EdgeSeq):
        self.name = name
        self.data = pd.DataFrame({
            'lang1': lang1_seq,
            'word1': word1_seq,
            'lang2': lang2_seq,
            'word2': word2_seq,
            'edge': edge_seq
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

    def get_cognates_in(self, lang: Lang) -> Set[Token]:
        if self.has_lang(lang):
            return self._lang2tokens[lang]
        else:
            return set()


@dataclass(eq=True, frozen=True)
class Coverage:
    num_covered: int
    total: int
    covered: List[Token] = field(repr=False)
    not_covered: List[Token] = field(repr=False)
    ratio: float = field(init=False)

    def __post_init__(self):
        self.ratio = float(f'{(self.num_covered / self.total):.3f}')


_EdgeDict = Dict[Token, _EdgeWithSource]


class TokenNotFound(Exception):
    """Raise this error if a token is not found in the graph."""


class EtymologicalGraph:
    """A graph representing the etymological relationship."""

    def __init__(self):
        self.tokens: Set[Token] = set()  # Stores all tokens
        self.out_edges: Dict[Token, _EdgeDict] = defaultdict(lambda: defaultdict(set))  # Stores all out-going edges
        self.in_edges: Dict[Token, _EdgeDict] = defaultdict(lambda: defaultdict(set))  # Stores all in-going edges.
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
            in_edges = {u: edge for u, edge in self.in_edges[v].items() if (
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

    def add_cognate_pair(self, token1: Token, token2: Token, source: str, edge: _Edge):
        self._check_finalized(False)
        if token1 != token2:  # NOTE(j_luo) Avoid self loops.
            self.tokens.add(token1)
            self.tokens.add(token2)
            edge = _EdgeWithSource(edge, source)
            self.out_edges[token1][token2].add(edge)
            self.in_edges[token2][token1].add(edge)

    def add_etymological_dictionary(self, ety_dict: EtymologicalDictionary):
        for _, row in ety_dict.data.iterrows():
            self.add_cognate_pair(row['word1'], row['word2'], ety_dict.name, row['edge'])

    def get_cognate_sets(self, token: Token) -> List[_CognateSet]:
        self._check_finalized(True)
        self._check_exists(token)
        return self._token2cog_sets[token]

    def _check_exists(self, token: Token):
        if token not in self.tokens:
            raise TokenNotFound(f'{token!r} not found in the graph.')

    def __getitem__(self, token: Token) -> Tuple[_EdgeDict, _EdgeDict]:
        self._check_exists(token)
        return (self.out_edges[token], self.in_edges[token])

    def has_cognates_in(self, token: Token, lang: Lang) -> bool:
        """Return whether `token` has a cognate in `lang`."""
        try:
            cog_sets = self.get_cognate_sets(token)
            return any(cog_set.has_lang(lang) for cog_set in cog_sets)
        except TokenNotFound:
            return False

    def compute_coverage(self, tokens: Iterable[Token], lang: Lang) -> Coverage:
        total = len(tokens)
        covered = list()
        not_covered = list()
        for token in tokens:
            is_covered = self.has_cognates_in(token, lang)
            if is_covered:
                covered.append(token)
            else:
                not_covered.append(token)
        num_covered = len(covered)
        return Coverage(num_covered, total, covered, not_covered)

    def get_cognates_in(self, token: Token, lang: Lang) -> Set[Token]:
        """Get cognates for `token` in language `lang`."""
        try:
            cog_sets = self.get_cognate_sets(token)
            ret = set()
            for cog_set in cog_sets:
                if cog_set.has_lang(lang):
                    ret.update(cog_set.get_cognates_in(lang))
            return ret
        except TokenNotFound:
            return set()


def get_ety_dict(table, column: str, name: str):
    """Get an EtymologicalDictionary instance."""
    token_col = f'{column}_tokens'
    # Default to the lemma language if lang_codes is not present.
    df = table[['Lemma', 'Sprachen', 'lang_codes', 'is_ref', 'is_single_ref', 'is_prefixed', token_col]]
    df['lang_codes'] = df[['Sprachen', 'lang_codes']].apply(lambda item: item[1] if item[1] else [item[0]], axis=1)
    df = df.reset_index(drop=True).explode(token_col)
    df = df.reset_index(drop=True).explode('lang_codes')
    df = df.dropna()
    df = df[(~df['is_ref']) | (df['is_single_ref']) | df['is_prefixed']]

    # def select_token(item: Tuple[bool, bool, Token]):
    #     is_single_ref, is_prefixed, token = item
    #     if is_single_ref:
    #         return token.tokens[0]
    #     # if is_prefixed:
    #     #     return token.tokens[1]
    #     return token

    # df[token_col] = df[['is_single_ref', 'is_prefixed', token_col]].apply(select_token, axis=1)

    # def get_rel_type(item: Tuple[bool, bool]) -> str:
    #     is_ref, is_prefixed = item
    #     if is_prefixed:
    #         return 'prefixed'
    #     if is_ref:
    #         return 'ref'
    #     return 'cog'

    # rel_type = df[['is_ref', 'is_prefixed']].apply(get_rel_type, axis=1)
    def get_token_and_edge(item: Tuple[Union[Token, MergedToken], bool, bool]) -> Tuple[Token, _Edge]:
        token, is_single_ref, is_prefixed = item
        info = None
        if is_single_ref:
            rel_type = RelType.SINGLE_REF
            token = token.tokens[0]
        elif is_prefixed:
            rel_type = RelType.PREFIXED
            info = token
            token = token.tokens[1]
        else:
            rel_type = RelType.COGNATE
        return token, _Edge(rel_type, info)

    token_and_edge = df[[token_col, 'is_single_ref', 'is_prefixed']].apply(get_token_and_edge, axis=1)
    df[token_col], df['edge'] = zip(*token_and_edge)

    lang1_seq = df['Sprachen']
    word1_seq = df['Lemma']
    lang2_seq = df['lang_codes']
    word2_seq = df[token_col]
    edge_seq = df['edge']

    if column == 'E':
        lang1_seq, lang2_seq = lang2_seq, lang1_seq
        word1_seq, word2_seq = word2_seq, word1_seq

    ety_dict = EtymologicalDictionary(name, lang1_seq, word1_seq, lang2_seq, word2_seq,
                                      edge_seq)
    return ety_dict


def get_ety_dict_from_tsv(tsv_path: str, column: str, name: str):
    table = process_table(tsv_path, column)
    ety_dict = get_ety_dict(table, column, name)
    return ety_dict
