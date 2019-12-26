from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from functools import lru_cache
from itertools import chain
from pathlib import Path
from typing import (Container, Dict, Iterable, List, NewType, Optional,
                    Sequence, Set, Tuple, TypeVar, Union)

import pandas as pd

from dev_misc.utils import Singleton, concat_lists, deprecated

CONVERT_HWAIR = True
KEEP_AFFIX = True
KEEP_DOUBTFUL = True  # False

Lang = NewType('Lang', str)


def load_bible(fname: str) -> pd.DataFrame:
    """Function to load the bible from a file."""
    data = list()
    with open(fname, 'r', encoding='utf8') as fin:
        while True:
            line = fin.readline()
            segs = line.strip().split('\t')
            if 'orig_chapter' in segs:
                columns = segs[1:]
                break

        for line in fin:
            data.append(tuple(line.strip().split('\t')))
    df = pd.DataFrame(data, columns=columns)
    df = df[['orig_book_index', 'orig_chapter', 'orig_verse', 'text']]
    df = df.rename(columns={
        'orig_book_index': 'book',
        'orig_chapter': 'chapter',
        'orig_verse': 'verse'
    })
    df = df.dropna().reset_index(drop=True)
    return df


def tokenize(bible: pd.DataFrame, lang: str, out_path: str, tokenizer_path='/scratch/j_luo/toolkit/mosesdecoder/scripts/tokenizer/tokenizer.perl'):
    """Tokenize the bible text. Expect the result from `load_bible` call as input."""
    with tempfile.NamedTemporaryFile('w+', encoding='utf8') as ftmp:
        for line in bible['text'].values:
            ftmp.write(line + '\n')

        # NOTE(j_luo) Do NOT forget to rewind.
        ftmp.seek(0)
        subprocess.run(
            f'perl {tokenizer_path} -threads 8 -no-escape -l {lang} < {ftmp.name} > {out_path}', shell=True, check=True)


def get_proto_dict(dict_path: str):
    """Get proto dictionary. This is hardcoded to only deal with German and Gothic right now."""
    proto_dict = pd.read_csv(dict_path)
    dfs = list()
    for idx, group in proto_dict.groupby('COGID'):
        langs = set(group['LANGUAGE'])
        if 'de' in langs and 'got' in langs:
            either_in = (group['LANGUAGE'] == 'de') | (group['LANGUAGE'] == 'got')
            df = group[either_in]
            dfs.append(df)
    df = pd.concat(dfs)
    pt = pd.pivot_table(df, index='COGID', columns='LANGUAGE', values='WORD', aggfunc=lambda l: l)
    pt['len_diff'] = (pt['de'].str.len() - pt['got'].str.len()).abs()
    to_drop = (pt['de'].str.contains(r'[-=]') | pt['got'].str.contains(r'[-=]'))
    # NOTE(j_luo) If difference in length > 3, we drop it.
    to_drop = to_drop | (pt['len_diff'] > 3)
    pt = pt[~to_drop].reset_index(drop=True)
    pt['de'] = pt['de'].str.replace('*', '', regex=False)
    pt['got'] = pt['got'].str.replace('*', '', regex=False)
    pt = pt.dropna()
    return pt


def get_vocab_and_words(path: str, lang: Lang):
    """Get vocabulary (a set) and words (a list) from a text file."""
    vocab = set()
    all_words = list()
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin:
            words = line.strip().split()
            tokens = list()
            for word in words:
                try:
                    token = get_token(word, lang)
                    tokens.append(token)
                except (InvalidString, AffixDiscarded, DoubtfulString):
                    pass
            vocab.update(tokens)
            all_words.extend(tokens)
    return vocab, all_words


_vorwort_abbr2note = dict()
_vorwort_note2abbr = dict()


def load_vorwort(vorwort_path: str):
    """Load vorwort so that subsequent procedures can proceed."""
    with open(vorwort_path, 'r', encoding='utf8') as fin:
        for line in fin:
            abbr, note = line.strip().split(' = ')
            note = f'#@!{note}!@#'
            _vorwort_abbr2note[abbr] = note
            _vorwort_note2abbr[note] = abbr


_got_prefixes = set()


def load_got_prefixes(prefix_path: str):
    with open(prefix_path, 'r', encoding='utf8') as fin:
        for line in fin:
            _got_prefixes.add(get_token(line.strip(), 'got'))


_note_group_pattern = re.compile(re.escape('#@!') + r'((?!' + re.escape('#@!') + r').)+')
_strip_pattern = re.compile(r'[\s,]*$')
_langs_to_keep = {
    'germanisch', 'indogermanisch', 'griechisch', 'lateinisch',
    'altenglisch', 'althochdeutsch', 'altalemannisch',
    'altbayerisch', 'altfriesisch', 'altfränkisch', 'altisländisch',
    'altmittelfränkisch', 'altnordisch', 'altniederdeutsch',
    'altniederfränkisch', 'altostniederfränkisch',
    'altrheinfränkisch', 'altsächsisch', 'altschwedisch',
    'altsüdmittelfränkisch', 'altsüdrheinfränkisch',
    'altrheinfränkisch', 'burgundisch', 'krimgotisch',
    'langobardisch', 'mittelgriechisch', 'mittelhochdeutsch',
    'mittellateinisch', 'mittelniederdeutsch', 'neuenglisch',
    'neuhochdeutsch', 'neuisländisch', 'runisch',
    'swebisch', 'vulgärlateinisch', 'westgotisch',
    'westgermanisch', 'gotisch'
}
_ref = ['siehe', 'siehe unter']
_compare = ['vergleiche']
_lang_to_keep = {
    f'#@!{note}!@#'
    for note in _langs_to_keep
}
_ref_notes = {
    f'#@!{note}!@#'
    for note in _ref
}
_compare_notes = {
    f'#@!{note}!@#'
    for note in _compare
}


def _strip(s: str) -> str:
    return re.sub(_strip_pattern, '', s)


@dataclass
class _Note:
    text: str
    is_ref: bool


def _split_note_groups(item: Tuple[Lang, str]) -> List[_Note]:
    """Split each segment into notes."""
    notes = list()
    lemma_lang, text = item
    for segment in text.split(';'):
        segment = segment.strip()
        matches = list(re.finditer(_note_group_pattern, segment))
        last_group = list()
        for match in matches:
            group = _strip(match.group())
            last_group.append(group)
            # This means this note group has something captured in addition to the note itself.
            if not group.endswith('!@#'):
                notes.append(' '.join(last_group))
                last_group = list()
        if last_group:
            notes.append(' '.join(last_group))
    ret = list()
    for note in notes:
        lang_relevant = any(x in note for x in _lang_to_keep)
        has_ref = any(x in note for x in _ref_notes)
        has_compare = any(x in note for x in _compare_notes)
        # Whenever "vergleiche" starts, break the loop.
        if has_compare:
            break
        is_ref = has_ref and (not lang_relevant or _vorwort_abbr2note[lemma_lang + '.'] in note)
        if lang_relevant or has_ref:
            ret.append(_Note(note, is_ref))

    return ret


def _split_into_tokens(item: Tuple[str, Lang]) -> List[Token]:
    """Split the lemma column into tokens."""
    s, lang = item
    ret = list()
    for t in s.split(','):
        try:
            token = get_token(t, lang)
            ret.append(token)
        except (AffixDiscarded, InvalidString, DoubtfulString, InvalidHyphens):
            pass
    return list(set(ret))


def _remove_invalid(note: _Note) -> _Note:
    """Remove invalid patterns found in the E/W columns.

    Note that once an invalid pattern is found, everything following it is removed as well.
    """
    # If there is a capitalized phrase (of more than one word) within any group (as a result of split by comma),
    # most likely it belongs to some note or reference.
    # For instance, "Lehmann B8", "Feist 462" and "Bedeutung dunkel".
    text = note.text
    text = re.sub(note_pattern, '', text)
    text = re.sub(r'[A-Z][\w]*\s+[\w\s]+.*', '', text)
    # Capitalized words are most likely to be some typo for notes provided in vorwort.
    # text = re.sub(r'\b[A-Z]\w*\b.*', '', text)
    # # Short and capitalized words (length <= 3) are most likely to be some typo for notes provided in vorwort.
    # s = re.sub(r'\b[A-Z]\w{0,2}\b.*', '', s)
    # # Short and capitalized words (length <= 3) are most likely to be some typo for notes provided in vorwort.
    # s = re.sub(r'\b[A-Z]\w{0,2}\b.*', '', s)
    # # Digits are removed.
    # s = re.sub(r'\b\d+\b.*', '', s)
    return _Note(text, note.is_ref)


_langs_to_keep_pattern = {
    lang: re.escape('#@!') + lang + re.escape('!@#')
    for lang in _langs_to_keep
}


def _get_lang_codes(note: _Note) -> List[str]:
    text = note.text
    ret = list()
    for lang in _langs_to_keep:
        if re.search(_langs_to_keep_pattern[lang], text):
            code = _vorwort_note2abbr['#@!' + lang + '!@#'].strip('.')
            ret.append(code)
    if len(ret) > 1:
        logging.warning(f'More than one language is extracted from a single note. Discarding this row: \n{text!r}')
        return list()
    return ret


note_pattern = re.compile(re.escape('#@!') + r'.+?' + re.escape('!@#'))


def _get_col_tokens(item: Tuple[_Note, Lang, List[Lang]]) -> List[Token]:
    """Based on the column and language codes, extract actual tokens."""
    note, default_lang, lang_codes = item
    text = note.text

    lang = lang_codes[0] if lang_codes else default_lang
    ret = list()
    for t in text.split(','):
        try:
            t = t.strip()
            if t:
                token = get_token(t, lang)
                ret.append(token)
        except (InvalidString, InvalidHyphens, DoubtfulString):
            pass
    return ret


def _merge_morphemes(item: Tuple[_Note, List[Token]]) -> Union[List[Token], MergedToken]:
    # """Merge potential morphemes together.

    # Note that sometimes the morphemes are just variants, so two heuristics are used to make sure only
    # morphemes of proper lengths are merged:
    # 1. Length makes sense
    # 2. The starting characters make sense.
    # """
    """Merge morphemes when the note is a reference."""
    # merged = sum(morphemes, None)
    # merged.sense_idx = lemma.sense_idx

    # def diff(x, y):
    #     return abs(len(x) - len(y))

    # start_with_same_char = all(morphemes[0][0] == morpheme[0] for morpheme in morphemes)
    # if start_with_same_char or diff(lemma, merged) > min(diff(lemma, morpheme) for morpheme in morphemes):
    #     return morphemes
    # else:
    #     return [merged]
    note, morphemes = item
    if note.is_ref:
        return merge_tokens(morphemes)
    else:
        return morphemes


def process_table(tsv_path: str, column: str) -> pd.DataFrame:
    """Main function for processing a saved tsv table."""
    if column not in ['W', 'E']:
        raise ValueError(f'Column can only be W or E.')
    abbr_sub_func = _get_sub_func(_vorwort_abbr2note, full_word=True)

    # Load.
    table = pd.read_csv(tsv_path, sep='\t')
    table = table[['Lemma', 'Sprachen', column]].dropna().reset_index(drop=True)

    # # Expand the column so that each row corresponds to only one segment (i.e., piece of etymological information separated by ;).
    # table[column] = table[column].apply(lambda x: x.split(';'))
    # table = table.explode(column)

    # Organize the column so that each segment is split into groups (separated by different notes).
    table[column] = table[column].apply(abbr_sub_func)
    table[column] = table[['Sprachen', column]].apply(_split_note_groups, axis=1)
    table = pd.pivot_table(table.reset_index(),
                           index=['index', 'Lemma', 'Sprachen'], values=column, aggfunc=concat_lists)
    table = table.reset_index([1, 2]).explode(column)
    table = table.dropna()

    # Each piece of note is now cleaned, and language codes and actual tokens are extracted from each note.
    table['lang_codes'] = table[column].apply(_get_lang_codes)
    table[column] = table[column].apply(_remove_invalid)
    token_col = f'{column}_tokens'
    table[token_col] = table[[column, 'Sprachen', 'lang_codes']].apply(_get_col_tokens, axis=1)

    # Remove rows with empty notes.
    table = table[table[token_col].apply(len) > 0]
    table = table.reset_index(drop=True)

    # Lemmas are expanded by split into tokens.
    table['Lemma'] = table[['Lemma', 'Sprachen']].apply(_split_into_tokens, axis=1)
    table = table.explode('Lemma').dropna()

    # Merge morphemes.
    table[token_col] = table[[column, token_col]].apply(_merge_morphemes, axis=1)
    table['is_ref'] = table[column].apply(lambda note: note.is_ref)
    table['is_single_ref'] = table[token_col].apply(lambda t: isinstance(t, MergedToken) and t.is_single_reference)
    table['is_prefixed'] = table[token_col].apply(lambda t: isinstance(t, MergedToken) and t.is_prefixed)

    return table

# -------------------------------------------------------------- #
#             Main classes for representation the tokens.             #
# -------------------------------------------------------------- #


@unique
class TokenType(Enum):
    """This indicates the type of the token. This is useful when a raw string is provided, but without information
    about the morphemes (i.e., gas-light has two stems while gas-es has one stem and one suffix).

    Note that this is different from MorphemeType which indicates the type of the morpheme.
    """
    NORMAL = auto()
    SEGMENTED = auto()
    PREFIX = auto()
    SUFFIX = auto()
    INFIX = auto()
    INVALID = auto()

    @classmethod
    def is_affix(cls, token_type: TokenType) -> bool:
        return token_type in [TokenType.PREFIX, TokenType.SUFFIX, TokenType.INFIX]


@unique
class MorphemeType(Enum):
    UNKNOWN = auto()  # NOTE(j_luo) Use this when morpheme type is not provided.
    STEM = auto()
    PREFIX = auto()
    SUFFIX = auto()
    INFIX = auto()


def _remove_capitalization(s: str):
    if s[0].isupper():
        return s[0].lower() + s[1:]
    else:
        return s


def _sort_keys(map_table: Dict) -> List[str]:
    return sorted(map_table.keys(), key=lambda s: len(s), reverse=True)


def _get_sub_func(map_table: Dict[str, str], full_word: bool = False):
    """See
    https://stackoverflow.com/questions/2400504/easiest-way-to-replace-a-string-using-a-dictionary-of-replacements,
    https://stackoverflow.com/questions/20089922/python-regex-engine-look-behind-requires-fixed-width-pattern-error.
    """
    pattern_str = '|'.join(_sort_keys(map_table))
    pattern_str = '|'.join(re.escape(key) for key in _sort_keys(map_table))
    if full_word:
        pattern_str = r'(?<!\w)(' + pattern_str + r')(?=\W|$)'
    pattern = re.compile(pattern_str)

    def sub(s):
        return pattern.sub(lambda x, map_table=map_table: map_table[x.group()], s)

    return sub


_standardize_map = {
    'hw': 'ƕ',
    'hv': 'ƕ'
}
_standardize_sub = _get_sub_func(_standardize_map)


class DoubtfulString(Exception):
    """Raise this error if "?" is in the string and `KEEP_DOUBTFUL` is False."""


def _standardize(s: str, lang: Lang) -> Tuple[str, Optional[int]]:  # , Optional[str]]:
    """Standardize the string in a conservative fashion. This is different from canonicalization."""
    if not KEEP_DOUBTFUL and '?' in s:
        raise DoubtfulString(f'Doubtful string {s!r}.')
    try:
        sense_idx = re.search(r'\s\((\d+)\)', s).group(1)
    except AttributeError:
        sense_idx = None

    # Replace digits/#/*/parentheses/brackets/question marks/equal signs with whitespace.
    s = re.sub(r'[?*\d#\[\]=]', '', s)
    # Anything inside parentheses are removed (including the parentheses).
    s = re.sub(r'\(.*?\)', '', s)

    s = re.sub(r'\s+', ' ', s)
    s = s.strip()

    # Convert hw/hv to ƕ if specified and the language is got.
    if lang == 'got' and CONVERT_HWAIR:
        s = _standardize_sub(s)
    return s, sense_idx  # , trans_s


_deaccent_map = {
    'á': 'a',
    'ï': 'i',
    'ā': 'a',
    'ū': 'u',
    'ē': 'e',
    'ō': 'o',
    'ī': 'i',
    'í': 'i',
    'ú': 'u'
}
_deaccent_sub = _get_sub_func(_deaccent_map)


def _canonicalize(s: str, lang: Lang) -> str:
    """Further canonicalize the string.

    This is more aggressive than standardization. Should be called only after validating the string.
    Also, the length of the string should not be changed.
    """
    orig_len = len(s)
    # Remove capitalization.
    s = _remove_capitalization(s)
    # De-accent for Gothic.
    if lang == 'got':
        s = _deaccent_sub(s)
    assert len(s) == orig_len
    return s


class InvalidString(Exception):
    """Invalid string."""


_to_check = {'_', '.', '·', ' '}


def _validate(s: str):
    """Validate that the string is not empty and doesn't contain weird characters."""
    if not s:
        raise InvalidString('Empty string.')
    violations = sorted(_to_check & set(s))
    if violations:
        raise InvalidString(f'Raw string {s!r} not valid, containing {violations!r}.')


class AdditionIncompatible(Exception):
    """Raise this error when adding two tokens of incompatible types. For instance, adding two prefixes together."""


class Token:

    def __init__(self,
                 lang: Lang,
                 raw_morphemes: List[str],
                 canonical_morphemes: List[str],
                 morpheme_types: List[MorphemeType],
                 sense_idx: Optional[int] = None):
        self.lang = lang
        self.raw_morphemes = raw_morphemes
        self.canonical_morphemes = canonical_morphemes
        self.morpheme_types = morpheme_types
        self.canonical_string = ''.join(self.canonical_morphemes)
        self.sense_idx = sense_idx

    def __str__(self):
        return self.canonical_string

    def __len__(self):
        return len(str(self))

    def __repr__(self):
        fields = list()
        if self.sense_idx is None:
            fields.append(str(self))
        else:
            fields.append(f'{self}@{self.sense_idx}')
        fields.append(self.lang)
        return f'Token({", ".join(fields)})'

    @property
    def signature(self) -> Tuple[str, int, Lang]:
        return (str(self), self.sense_idx, self.lang)

    def __eq__(self, other: Token):
        if not isinstance(other, Token):
            return False
        return self.signature == other.signature

    def __lt__(self, other: Token):
        return self.signature < other.signature

    def __getitem__(self, idx: int) -> str:
        return str(self)[idx]

    def is_same_string(self, other: str):
        """This is different from __eq__ because it is compared against a string."""
        if not isinstance(other, str):
            raise TypeError(f'Can only call this against strings.')
        return str(self) == other

    def __hash__(self):
        return hash(self.signature)

    def __add__(self, other: Token) -> Token:
        token_factory = TokenFactory()
        return token_factory.add_tokens(self, other)

    def __radd__(self, other: Union[Token, None]) -> Token:
        if other is None:
            return self
        else:
            return self.__add__(other)

    def __iter__(self):
        yield from str(self)


class MergedToken:

    def __init__(self, tokens: Sequence[Token]):
        self.tokens = tokens

    def __str__(self):
        return '^'.join([str(token) for token in self.tokens])

    @property
    def is_single_reference(self):
        return len(self.tokens) == 1

    @property
    def is_prefixed(self):
        return len(self.tokens) == 2 and self.tokens[0] in _got_prefixes

    def __hash__(self):
        return hash(tuple(self.tokens))

    def __eq__(self, other: MergedToken):
        if not isinstance(other, MergedToken):
            return False
        return len(self.tokens) == len(other.tokens) and all(t1 == t2 for t1, t2 in zip(self.tokens, other.tokens))


def _get_token_type(s: str) -> TokenType:
    hyphen_matches = list(re.finditer('-', s))
    if len(hyphen_matches) == 0:
        return TokenType.NORMAL
    else:
        at_start = s.startswith('-')
        at_end = s.endswith('-')
        if len(hyphen_matches) == 1:
            if at_start:
                return TokenType.SUFFIX
            elif at_end:
                return TokenType.PREFIX
            else:
                return TokenType.SEGMENTED
        # No consecutive hyphens.
        for match0, match1 in zip(hyphen_matches[:-1], hyphen_matches[1:]):
            if match1.start() == match0.start() + 1:
                return TokenType.INVALID
        # When there are more than two hyphens, you should not have any hyphen starting or ending the string.
        if len(hyphen_matches) > 2:
            if not at_start and not at_end:
                return TokenType.SEGMENTED
            else:
                return TokenType.INVALID
        # When there are exactly two hyphens, you can either have one at each end (i.e., an infix), or none at either end.
        if at_start ^ at_end:
            return TokenType.INVALID
        elif at_start and at_end:
            return TokenType.INFIX
        else:
            return TokenType.SEGMENTED


class InvalidHyphens(Exception):
    """Invalid hyphenation."""


class AffixDiscarded(Exception):
    """Raise this error if affix should be dicarded."""


class TokenFactory(Singleton):

    _tokens: Dict[str, Token] = dict()

    def add_tokens(self, token1: Token, token2: Token) -> Token:
        if not isinstance(token1, Token) or not isinstance(token2, Token):
            raise TypeError(f'Must add instances of Token (or its subclass).')
        if token1.lang != token2.lang:
            raise RuntimeError(f'Cannot add two tokens of different languages.')

        raw_morphemes = token1.raw_morphemes + token2.raw_morphemes
        canonical_morphemes = token1.canonical_morphemes + token2.canonical_morphemes
        morpheme_types = token1.morpheme_types + token2.morpheme_types
        return Token(token1.lang, raw_morphemes, canonical_morphemes, morpheme_types)

    def merge_tokens(self, tokens: Sequence[Token]) -> MergedToken:
        return MergedToken(tokens)

    def get_token(self, raw_string: str, lang: Lang) -> Token:
        cls = type(self)
        key = (raw_string, lang)
        if key in cls._tokens:
            token = cls._tokens[key]
        else:
            # standard_string, sense_idx, trans_s = _standardize(raw_string, lang)
            standard_string, sense_idx = _standardize(raw_string, lang)
            # Deal with hyphens.
            token_type = _get_token_type(standard_string)
            if token_type == TokenType.INVALID:
                raise InvalidHyphens(f'Weird hyphens in string {standard_string!r}.')
            if TokenType.is_affix(token_type) and not KEEP_AFFIX:
                raise AffixDiscarded(f'Affixes are configured to be discarded.')

            raw_morphemes = standard_string.strip('-').split('-')
            # NOTE(j_luo) Validate the string after removing hyphens -- to deal with some weird string like " -"..
            _validate(''.join(raw_morphemes))
            canonical_morphemes = [_canonicalize(morpheme, lang) for morpheme in raw_morphemes]
            if TokenType.is_affix(token_type):
                morpheme_types = [MorphemeType[token_type.name]]
            else:
                # NOTE(j_luo) You can't know the morpheme types if it's not explicitly segmented.
                morpheme_types = [MorphemeType.UNKNOWN] * len(raw_morphemes)
            assert len(raw_morphemes) == len(canonical_morphemes) == len(morpheme_types)

            token = Token(lang, raw_morphemes, canonical_morphemes, morpheme_types,
                          sense_idx=sense_idx)
            self._tokens[raw_string] = token
        return token

    def clear_cache(self):
        """Remove all cached tokens."""
        cls = type(self)
        cls._tokens.clear()


get_token = TokenFactory().get_token
merge_tokens = TokenFactory().merge_tokens
clear_cache = TokenFactory().clear_cache

# -------------------------------------------------------------- #
#           Helpful classes for inspecting the corpus.           #
# -------------------------------------------------------------- #


class _Collection:
    """A collection is just a list of lexical units with some useful methods."""

    def __init__(self):
        self.data = list()

    def append(self, item):
        self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Collection: size ({len(self)})"


class TrackableCorpus:
    """A corpus that could track where each lexical unit is coming from (i.e., which sentence etc)."""

    def __init__(self, raw_data: Iterable[Iterable[str]]):
        self.directory: Dict[str, _Collection] = defaultdict(_Collection)
        for idx, item in enumerate(raw_data):
            for position, unit in enumerate(item):
                self.directory[unit].append((item, position))

    def __getitem__(self, unit: str) -> _Collection:
        return self.directory[unit]

    @property
    def all_units(self) -> Dict[str, int]:
        """Return all units tracked in the corpus and their corresponding counts."""
        return {k: len(self.directory[k]) for k in self.directory}
