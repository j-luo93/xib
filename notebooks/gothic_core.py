from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from enum import Enum, auto, unique
from pathlib import Path
from typing import (Container, Dict, Iterable, List, NewType, Optional,
                    Sequence, Tuple, TypeVar, Union)

import pandas as pd
from lxml.html import HTMLParser, fromstring
from pyquery import PyQuery as pq

from dev_misc.utils import Singleton, concat_lists

CONVERT_HWAIR = True
KEEP_AFFIX = True
KEEP_DOUBTFUL = False

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
                except (InvalidString, AffixDiscarded):
                    pass
            vocab.update(tokens)
            all_words.extend(tokens)
    return vocab, all_words


def count(iterables: Iterable[str], dictionaries: Iterable[Container[str]]):
    """Count how many words are covered by any of the provided dictionaries."""
    for iterable in iterables:
        total = len(iterable)
        covered = sum([any(w in dictionary for dictionary in dictionaries) for w in iterable])
        ratio = total / covered
        print(f'{total} / {covered} = {ratio:.3f}')

# -------------------------------------------------------------- #
#           Functions for dealing with wikiling tables.          #
# -------------------------------------------------------------- #


def get_table(filename: str) -> pd.DataFrame:
    """Get a table from one html file."""
    doc = pq(filename=filename, parser='html')
    table = doc('table#search-results')
    header = table('thead')
    assert len(header) == 1

    columns = [th.text for th in header('th')]

    rows = list()

    body = table('tbody')
    for tr in body('tr'):
        row = pq(tr)
        row = [td.text for td in row('td')]
        assert len(row) == len(columns)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def save_table(folder: str, out_path: str):
    """Get tables from all html files within one folder (e.g., `got`), and then save a concatenated wikiling table."""
    folder = Path(folder)
    tables = list()

    def get_page_num(path: Path) -> int:

        return int(re.search(r'page\.(\d+)\.html', str(path)).group(1))

    for filename in sorted(folder.glob('page.*.html'), key=get_page_num):
        table = get_table(filename)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table = table.reset_index(drop=True).drop(columns='#')
    table.to_csv(out_path, index=None, sep='\t')


_note_group_pattern = re.compile(re.escape('#@!') + r'((?!' + re.escape('#@!') + r').)+')
_strip_pattern = re.compile(r'[\s,]*$')
_lang2code = {
    'germanisch': 'gem',
    'indogermanisch': 'idg',
    'griechisch': 'el',
    'lateinisch': 'lat'
}
_ref = ['siehe', 'vergleiche', 'siehe unter']
_to_keep = {
    f'#@!{note}!@#'
    for note in list(_lang2code.keys()) + _ref
}


def _strip(s: str) -> str:
    return re.sub(_strip_pattern, '', s)


def _split_note_groups(s: str) -> List[str]:
    """Split each segment into notes."""
    matches = list(re.finditer(_note_group_pattern, s))
    ret = list()
    last_group = list()
    for match in matches:
        group = _strip(match.group())
        last_group.append(group)
        # This means this note group has something captured in addition to the note itself.
        if not group.endswith('!@#'):
            ret.append(' '.join(last_group))
            last_group = list()
    if last_group:
        ret.append(' '.join(last_group))
    ret = list(filter(lambda x: any(note in x for note in _to_keep), ret))
    return ret


def _split_into_tokens(s: str, lang: str) -> List[str]:
    """Split the lemma column into tokens."""
    ret = list()
    for t in s.split(','):
        try:
            token = get_token(t, lang)
            ret.append(token)
        except (AffixDiscarded, InvalidString, DoubtfulString):
            pass
    return list(set(ret))


def _remove_invalid(s: str) -> str:
    """Remove invalid patterns found in the E columns.

    Note that once an invalid pattern is found, everything following it is removed as well.
    """
    # Short and capitalized words (length <= 3) are most likely to be some typo for notes provided in vorwort.
    s = re.sub(r'\b[A-Z]\w{0,2}\b.*', '', s)
    # Digits are removed.
    s = re.sub(r'\b\d+\b.*', '', s)
    # If there is a phrase (of more than one word) within any group (as a result of split by comma),
    # most likely it belongs to some note or reference.
    # For instance, "Lehmann B8", "Feist 462" and "Bedeutung dunkel".
    s = re.sub(r'\w[\w]*\s+[\w\s]+.*', '', s)
    return s


_lang2code_pattern = {
    lang: re.escape('#@!') + lang + re.escape('!@#')
    for lang in _lang2code
}


def _get_lang_codes(s: str) -> List[str]:
    ret = list()
    for lang, code in _lang2code.items():
        if re.search(_lang2code_pattern[lang], s):
            ret.append(code)
    return ret


note_pattern = re.compile(re.escape('#@!') + r'.+?' + re.escape('!@#'))


def _get_E_tokens(item: Tuple[str, List[Lang]]) -> List[_BaseToken]:
    """Based on the E column and language codes, extract actual tokens."""
    s, lang_codes = item
    lang = 'unk' if lang_codes else 'got'
    s = re.sub(note_pattern, '', s)
    ret = list()
    try:
        for t in s.split(','):
            t = t.strip()
            if t:
                token = get_token(t, lang)
                ret.append(token)
    except (InvalidString, InvalidHyphens, DoubtfulString):
        ret = list()
    return ret


def _merge_morphemes(item: Tuple[str, List[_BaseToken]]) -> List[_BaseToken]:
    """Merge potential morphemes together.

    Note that sometimes the morphemes are just variants, so a length-based heuristic is used to make sure only
    morphemes of proper lengths are merged.
    """
    lemma, morphemes = item
    merged = sum(morphemes, None)

    def diff(x, y):
        return abs(len(x) - len(y))

    if diff(lemma, merged) > min(diff(lemma, morpheme) for morpheme in morphemes):
        return morphemes
    else:
        return [merged]


def process_table(tsv_path: str, vorwort_path: str) -> pd.DataFrame:
    """Main function for processing a saved wiking table."""
    abbr_map = dict()
    with open(vorwort_path, 'r', encoding='utf8') as fin:
        for line in fin:
            abbr, note = line.strip().split(' = ')
            note = f'#@!{note}!@#'
            abbr_map[abbr] = note
    abbr_sub_func = _get_sub_func(abbr_map, full_word=True)

    # Load.
    table = pd.read_csv(tsv_path, sep='\t')
    table = table[['Lemma', 'E']].dropna().reset_index(drop=True)

    # Lemmas are expanded by split into tokens.
    table['Lemma'] = table['Lemma'].apply(lambda s: _split_into_tokens(s, 'got'))
    table = table.explode('Lemma').dropna()

    # Expand the E column so that each row corresponds to only one segment (i.e., piece of etymological information separated by ;).
    table['E'] = table['E'].apply(lambda x: x.split(';'))
    table = table.explode('E')

    # Organize the E column so that each segment is further split into groups (separated by different notes).
    table['E'] = table['E'].apply(abbr_sub_func)
    table['E'] = table['E'].apply(_split_note_groups)
    table = pd.pivot_table(table.reset_index(), index=['index', 'Lemma'], values='E', aggfunc=concat_lists)
    table = table.reset_index(1).explode('E')
    table = table.dropna()

    # Each piece of note is now cleaned, and language codes and actual tokens are extracted from each note.
    table['E'] = table['E'].apply(_remove_invalid)
    table['lang_codes'] = table['E'].apply(_get_lang_codes)
    table['E_tokens'] = table[['E', 'lang_codes']].apply(_get_E_tokens, axis=1)

    # Remove rows with empty notes.
    table = table[table['E_tokens'].apply(len) > 0]

    # Merge morphemes.
    table['E_tokens'] = table[['Lemma', 'E_tokens']].apply(_merge_morphemes, axis=1)

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


def _standardize(s: str) -> str:
    """Standardize the string in a conservative fashion. This is different from canonicalization."""
    if not KEEP_DOUBTFUL and '?' in s:
        raise DoubtfulString(f'Doubtful string {s!r}.')
    # Replace digits/#/*/parentheses/brackets/question marks/equal signs with whitespace.
    s = re.sub(r'[?*\d#()\[\]=]', '', s)
    s = re.sub(r'\s', ' ', s)
    # Convert hw/hv to ƕ if specified.
    if CONVERT_HWAIR:
        s = _standardize_sub(s)
    s = s.strip()
    return s


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


class _BaseToken:
    """This represents a Token class that is notebook-friendly (pandas won't mess things up) since it doesn't define __iter__.

    More at https://github.com/pandas-dev/pandas/issues/22333.
    """

    def __init__(self, raw_morphemes: List[str], canonical_morphemes: List[str], morpheme_types: List[MorphemeType]):
        self.raw_morphemes = raw_morphemes
        self.canonical_morphemes = canonical_morphemes
        self.morpheme_types = morpheme_types

    def __str__(self):
        return ''.join(self.canonical_morphemes)

    def __len__(self):
        return len(str(self))

    def __repr__(self):
        return f'Token({self})'

    def __eq__(self, other: _BaseToken):
        if not isinstance(other, _BaseToken):
            return False
        return str(self) == str(other)

    def __lt__(self, other: _BaseToken):
        return str(self) < str(other)

    def __hash__(self):
        return hash(str(self))

    def __add__(self, other: _BaseToken) -> _BaseToken:
        token_factory = _TokenFactory()
        return token_factory.add_tokens(self, other)

    def __radd__(self, other: Union[_BaseToken, None]) -> _BaseToken:
        if other is None:
            return self
        else:
            return self.__add__(other)


class _Token(_BaseToken):

    def __iter__(self):
        yield from str(self)


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


NOTEBOOK_MODE: bool = False


def notebook_mode(flag: bool):
    """Set notebook mode on or off. This will change the token class."""
    global NOTEBOOK_MODE
    NOTEBOOK_MODE = flag


T = TypeVar('T', bound=_BaseToken)


class _TokenFactory(Singleton):

    _tokens: Dict[str, _Token] = dict()

    def add_tokens(self, token1: T, token2: T) -> T:
        if not isinstance(token1, _BaseToken) or not isinstance(token2, _BaseToken):
            raise TypeError(f'Must add instances of _BaseToken (or its subclass).')
        raw_morphemes = token1.raw_morphemes + token2.raw_morphemes
        canonical_morphemes = token1.canonical_morphemes + token2.canonical_morphemes
        morpheme_types = token1.morpheme_types + token2.morpheme_types
        return self.token_cls(raw_morphemes, canonical_morphemes, morpheme_types)

    @property
    def token_cls(self):
        return _BaseToken if NOTEBOOK_MODE else _Token

    def get_token(self, raw_string: str, lang: Lang) -> T:
        cls = type(self)
        if raw_string in cls._tokens:
            token = cls._tokens[raw_string]
        else:
            standard_string = _standardize(raw_string)
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

            token = self.token_cls(raw_morphemes, canonical_morphemes, morpheme_types)
            self._tokens[raw_string] = token
        return token


get_token = _TokenFactory().get_token

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


# -------------------------------------------------------------- #
#              Classes for etymological dictionaries             #
# -------------------------------------------------------------- #

Word = NewType('Word', str)
LangSeq = Sequence[Lang]
WordSeq = Sequence[Word]


class EtymologicalDictionary:
    """Represents a dictionary that stores etymological information.

    This follows a list of conventions:
    1. The information is represented by a 4-tuple:
            (lang1, word1, lang2, word2)
       This means that word1 in lang1 is "connected" with word2 in lang2 where a connection might
       mean borrowing or cognation. Note that it is not directional.
    2. These tuples are stored in a DataFrame instance. Column names are "lang1", "word1", "lang2" and "word2".
    """

    def __init__(self, lang1_seq: LangSeq, word1_seq: WordSeq, lang2_seq: LangSeq, word2_seq: WordSeq, ):
        self.data = pd.DataFrame({
            'lang1': lang1_seq,
            'word1': word1_seq,
            'lang2': lang2_seq,
            'word2': word2_seq
        })

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, lang1: str, lang2: str) -> EtymologicalDictionary:
        """Convert a normal data frame into an EtymologicalDictionoary instance.

        The input `df` must have columns corresponding to the languages in question. So for a Gothic-German dictionary,
        you need to have a `got` column and a `de` columns in `df`. Specifying `lang1` as `got` and `lang2` as `de` will
        convert `df` accordingly to an EtymologicalDictionary instance.
        """
        word1_seq = [get_token(w) for w in df[lang1]]
        word2_seq = [get_token(w) for w in df[lang2]]
        lang1_seq = [lang1] * len(word1_seq)
        lang2_seq = [lang2] * len(word2_seq)
        ety_dict = EtymologicalDictionary(lang1_seq, word1_seq, lang2_seq, word2_seq)
        return ety_dict


# -------------------------------------------------------------- #
#                  Convert html files into tsv.                  #
# -------------------------------------------------------------- #

reserved_seps = {
    'got': ['ÜG', 'ÜE', 'Vw', 'Hw', 'Q', 'I', 'E', 'B', 'R', 'L', 'Son'],
    'edel': ['Vw', 'Hw', 'Q', 'I', 'E', 'L', 'Son', 'GB', 'BM', 'F'],
    'germ': ['RB', 'Vw', 'Hw', 'Q', 'I', 'E', 'W', 'B', 'L', 'Son'],
    'idg': ['RB', 'ÜG', 'Vw', 'Hw', 'E', 'W', 'L', 'Son'],
    'ae': ['ÜG', 'Vw', 'Hw', 'Q', 'I', 'E', 'W', 'R', 'L', 'Son']
}


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
        prefix = f'{sep}\.:'
    return fr'({prefix}(?:{reserved_p}.)*;?\s*)?'


def get_match_regex(dataset: str) -> re.Pattern:
    lemma_p = r'^([^,]+),?\s*'
    alternative_p = r'((?:\b[^\.]+\b)(?!\.),?\s*)*'
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


group2col = {
    dataset: ['Lemma', 'Alternative', 'Sprachen', 'Wortarten_all', 'Bedeutungen_all'] + seps
    for dataset, seps in reserved_seps.items()
}


def remove_sep(prefix: str, s: str) -> str:
    """Remove separators."""
    if s is not None:
        return re.compile(fr'{prefix}\.:\s*([^;]*);?\s*').match(s).group(1)


def get_df_from_doc(doc: Iterable[str], dataset: str) -> pd.DataFrame:
    regex = get_match_regex(dataset)
    matches = list()
    for line in doc:
        matches.append(regex.match(line.strip()))

    # Extract columns from matches.
    data = list()
    for i, match in enumerate(matches):
        row = [match.group(idx) for idx, col in enumerate(group2col[dataset], 1)]
        data.append(tuple(row))

    df = pd.DataFrame(data, columns=group2col[dataset])

    for sep in reserved_seps[dataset]:
        df[sep] = df[sep].apply(lambda s: remove_sep(sep, s))

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
        for elem in chain(doc('p.MsoNormal'), doc('p.MsoPlainText')):
            entry = pq(elem)
            text = entry.text()
            text = re.sub(r'\s', ' ', text).strip()  # EDEL has many multi-line contents.
            if text:  # Some entries are empty.
                yield text
