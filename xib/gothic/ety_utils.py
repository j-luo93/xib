from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heapify, heappop
from typing import (Any, Dict, Iterable, List, NewType, Optional, Sequence,
                    Set, Tuple, Union)

import pandas as pd

from dev_misc.utils import cached_property
from xib.gothic.core import (InvalidString, Lang, MergedToken, Token,
                             get_token, process_table)
from xib.gothic.transliterator import MultilingualTranliterator

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
    info: Optional[MergedToken] = None


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
        edge_seq = [_Edge(RelType.COGNATE) for _ in range(len(word1_seq))]
        ety_dict = EtymologicalDictionary(name, lang1_seq, word1_seq, lang2_seq, word2_seq, edge_seq)
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


@dataclass
class Coverage:
    num_covered: int
    total: int
    covered: List[Token] = field(repr=False)
    not_covered: List[Token] = field(repr=False)
    ratio: float = field(init=False)

    def __post_init__(self):
        self.ratio = float(f'{(self.num_covered / self.total):.3f}')


_EdgeDict = Dict[Token, Set[_EdgeWithSource]]


class TokenNotFound(Exception):
    """Raise this error if a token is not found in the graph."""


@dataclass
class _PrefixStep:
    src_lang: Lang
    tgt_lang: Lang
    prefix: Token = None


@dataclass
class PrefixPath:
    path: List[_PrefixStep] = field(default_factory=list)

    @cached_property
    def is_empty(self) -> bool:
        for ps in self.path:  # pylint: disable=not-an-iterable
            if ps.prefix:
                return False
        return True


class PrefixInOtherLanguage(Exception):
    """Raise this error when a prefix is found in some other language rather than the reference language."""


class UncontiguousPrefix(Exception):
    """Raise this error when a chain of prefixes are found but they are not contiguous."""


class UnmatchedPrefix(Exception):
    """Raise this error when the prefix is not matched with the token."""


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
            edge = _EdgeWithSource(edge.rel_type, source, info=edge.info)
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

    def get_cognates_in(self, token: Token, lang: Lang) -> List[Tuple[_CognateSet, Set[Token]]]:
        """Get cognates for `token` in language `lang`."""
        try:
            cog_sets = self.get_cognate_sets(token)
            ret = list()
            for cog_set in cog_sets:
                tokens = set()
                if cog_set.has_lang(lang):
                    tokens = cog_set.get_cognates_in(lang)
                if tokens:
                    ret.append((cog_set, tokens))
            return ret
        except TokenNotFound:
            return list()

    def get_paths_to_closest_common_ancestor(self, token1: Token, token2: Token, cog_set: _CognateSet) -> Tuple[List[Token], List[Token]]:
        """Return paths to the closest common ancestory for `token1` and `token2`."""

        root = cog_set.root

        def get_path_to_root(token: Token, path: List[Token]):
            path.append(token)
            if token is root:
                return
            for u, edge in self.in_edges[token].items():
                if u in cog_set:
                    get_path_to_root(u, path)
                    break

        path1 = list()
        get_path_to_root(token1, path1)
        path2 = list()
        get_path_to_root(token2, path2)
        i = 0
        for i, (t1, t2) in enumerate(zip(path1[::-1], path2[::-1])):
            if t1 != t2:
                break
        if t1 == t2:
            i += 1
        path1 = path1[:len(path1) - i + 1]
        path2 = path2[:len(path2) - i + 1]
        return path1, path2

    def get_prefixes_along_paths(self, token1: Token, token2: Token, cog_set: _CognateSet) -> Tuple[PrefixPath, PrefixPath]:
        path1, path2 = self.get_paths_to_closest_common_ancestor(token1, token2, cog_set)

        def get_prefixes(path: List[Token]) -> PrefixPath:
            pp = PrefixPath()
            for u, v in zip(path[:-1], path[1:]):
                edges = self.in_edges[u][v]
                # if len(edges) > 1:
                #     logging.warning('u -> v has two different edges. Only one is kept.')
                ps = _PrefixStep(u.lang, v.lang)
                for edge in edges:
                    if edge.rel_type == RelType.PREFIXED:
                        prefix = edge.info.tokens[0]
                        if ps.prefix is not None and ps.prefix != prefix:
                            logging.warning(
                                f'Encountered two different prefixes for the same u -> v edge. Only keeping the last one.')
                        ps.prefix = prefix  # pylint: disable=no-member
                pp.path.append(ps)  # pylint: disable=no-member

            return pp

        prefixes1 = get_prefixes(path1)
        prefixes2 = get_prefixes(path2)

        return prefixes1, prefixes2

    def translate(self, token: Token, lang: Lang) -> Set[str]:
        cog_ret = self.get_cognates_in(token, lang)
        ret = list()
        src_lang = token.lang

        def reconstruct(pp, src, tgt, side):
            ref_lang = src.lang

            # Avoid having prefixes in any other language.
            for ps in pp.path:
                if ps.prefix and (ps.src_lang != ref_lang or ps.tgt_lang != ref_lang):
                    raise PrefixInOtherLanguage(
                        f'Reference lang is {ref_lang}, but got u -> v ({ps.src_lang} -> {ps.tgt_lang}) with prefix {ps.prefix!r}.')

            # The chain of prefixes can only be a contiguous chain from 0-th position.
            pos = list()
            for i, ps in enumerate(pp.path):
                if ps.prefix:
                    pos.append(i)
            for i, p in enumerate(pos):
                if i != p:
                    raise UncontiguousPrefix(f'Uncontiguos chain of prefixes are found in {pp}.')

            translation = ''
            text = str(src) if side == 'src' else str(tgt)
            for p in pos:
                prefix = pp.path[p].prefix
                if text.startswith(str(prefix)):
                    text = text[len(prefix):]
                    translation += f'[{"?" * len(prefix)}]' if side == 'src' else f'[{prefix}]'
                else:
                    raise UnmatchedPrefix(f'Prefix {prefix!r} not matched with the token {token!r}.')

            translation += str(cognate)

            return translation

        def safe_add_translation(pp, src, tgt, side):
            try:
                translation = reconstruct(pp, src, tgt, side)
                ret.append(translation)
            except (PrefixInOtherLanguage, UncontiguousPrefix, UnmatchedPrefix) as e:
                pass

        for cog_set, cognates in cog_ret:
            for cognate in cognates:
                pref1, pref2 = self.get_prefixes_along_paths(token, cognate, cog_set)
                # We don't know how to deal with them yet.
                if not pref1.is_empty and not pref2.is_empty:
                    #                 logging.warning(f'Skip {token!r} and {cognate!r} since both sides have prefixes.\n{pref1} and {pref2}')
                    continue
                if not pref1.is_empty:
                    safe_add_translation(pref1, token, cognate, 'src')
                elif not pref2.is_empty:
                    safe_add_translation(pref2, token, cognate, 'tgt')
                else:
                    ret.append(str(cognate))
        return set(ret)

    def translate_word_lemma_pair(self, word: str, lemma: str, src_lang: Lang, tgt_lang: Lang) -> Tuple[Set[str], Set[str]]:
        try:
            word_translations = self.translate(get_token(word, src_lang), tgt_lang)
        except InvalidString:
            return set(), set()
        lemma_translations = self.translate(get_token(lemma.replace('-', ''), src_lang), tgt_lang)
        stem_translations = set()
        if '-' in lemma:
            prefix, *stem = lemma.split('-')
            stem = ''.join(stem)
            try:
                stem_translations = self.translate(get_token(stem, src_lang), tgt_lang)
                stem_translations = {f'[{"?" * len(prefix)}]' + str(cog) for cog in stem_translations}
            except InvalidString:
                pass
        return word_translations, lemma_translations | stem_translations

    def translate_conll(self, in_path: str, out_path: str, src_lang: Lang, tgt_lang: Lang, transliterator: MultilingualTranliterator):
        word_cnt = 0
        word_covered = 0
        vocab = set()
        vocab_covered = 0

        def safe_get_standardize_str(s: str) -> str:
            try:
                return str(get_token(s, src_lang))
            except InvalidString:
                return s

        def union_all(sets: Sequence[Set]) -> Set:
            ret = set()
            for s in sets:
                ret.update(s)
            return ret

        with open(in_path, 'r', encoding='utf8') as fin, open(out_path, 'w', encoding='utf8') as fout:
            for line in fin:
                line = line.strip()
                if line:
                    word, lemma = line.split('\t')[1:3]
                    word_cnt += 1

                    word_translations, lemma_translations = self.translate_word_lemma_pair(
                        word, lemma, src_lang, tgt_lang)
                    if word_translations or lemma_translations:
                        word_covered += 1
                    if word not in vocab:
                        vocab.add(word)
                        if word_translations or lemma_translations:
                            vocab_covered += 1
                    wt = ','.join(word_translations)
                    lt = ','.join(lemma_translations)
                    w = safe_get_standardize_str(word)
                    l = safe_get_standardize_str(lemma)

                    try:
                        wt_ipa = ','.join(map(str, union_all(transliterator.transliterate(_wt, tgt_lang)
                                                             for _wt in word_translations)))
                        lt_ipa = ','.join(map(str, union_all(transliterator.transliterate(_lt, tgt_lang)
                                                             for _lt in word_translations)))
                        w_ipa = ','.join(map(str, transliterator.transliterate(w, src_lang)))
                        l_ipa = ','.join(map(str, transliterator.transliterate(l, src_lang)))
                        fout.write('|'.join([w, w_ipa, l, l_ipa, wt, wt_ipa, lt, lt_ipa]) + ' ')
                    except ValueError as e:
                        logging.exception(e)
                else:
                    fout.write('\n')
        word_coverage = word_covered / word_cnt
        vocab_coverage = vocab_covered / len(vocab)
        print(f'{word_covered} / {word_cnt} = {word_coverage:.3f}')
        print(f'{vocab_covered} / {len(vocab)} = {vocab_coverage:.3f}')


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
