from __future__ import annotations

import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import (IO, ClassVar, Dict, FrozenSet, Iterable, Iterator, List,
                    Optional, Sequence, Set, Tuple, TypeVar, Union)

import numpy as np
import pandas as pd
import torch

from dev_misc import add_argument, g
from dev_misc.devlib import BaseBatch, batch_class, get_array, get_length_mask
from dev_misc.devlib.named_tensor import NoName, Rename
from dev_misc.utils import (Singleton, cached_property, check_explicit_arg,
                            union_sets)
from xib.aligned_corpus.char_set import CharSet, CharSetFactory
from xib.aligned_corpus.ipa_sequence import Content, IpaSequence
from xib.aligned_corpus.transcriber import MultilingualTranscriber
from xib.batch import convert_to_dense


@dataclass(eq=True, frozen=True)
class Word:
    """A word with form and ipa."""
    lang: str
    form: str
    ipa: FrozenSet[IpaSequence]

    @property
    def form_length(self) -> int:
        return len(self.form)

    @property
    def ipa_length(self) -> int:
        assert len(self.ipa) == 1
        return len(self.main_ipa)

    def __repr__(self):
        ipa_str = ','.join(map(str, self.ipa))
        return f'{self.lang};{self.form};{{{ipa_str}}}'

    @classmethod
    def from_saved_string(cls, saved_string: str) -> Word:
        lang, form, ipa_str = saved_string.split(';')
        ipa = frozenset({IpaSequence(s) for s in ipa_str[1:-1].split(',')})
        return cls(lang, form, ipa)

    @property
    def main_ipa(self) -> IpaSequence:
        # HACK(j_luo) Only take one of them.
        return list(self.ipa)[0]


@dataclass(eq=True, frozen=True)
class Stem(Word):
    """A stem is just a word with start and end."""
    start: int
    end: int
    full_form: str

    def __repr__(self):
        ipa_str = ','.join(map(str, self.ipa))
        return f'{self.lang};{self.form}:{self.start}:{self.end}:{self.full_form};{{{ipa_str}}}'

    @classmethod
    def from_saved_string(cls, saved_string: str) -> Word:
        lang, form_with_orig, ipa_str = saved_string.split(';')
        form, start, end, full_form = form_with_orig.split(':')
        start = int(start)
        end = int(end)
        ipa = frozenset({IpaSequence(s) for s in ipa_str[1:-1].split(',')})
        return cls(lang, form, ipa, start, end, full_form)


_Signature = Tuple[str, str]


class WordFactory(Singleton):

    _cache: ClassVar[Dict[_Signature, Word]] = dict()

    def get_word(self, lang: str, form: str, transcriber: MultilingualTranscriber) -> Word:
        cls = type(self)
        key = (lang, form)
        if key in cls._cache:
            return cls._cache[key]

        ipa_strings = transcriber.transcribe(form, lang)
        ipa = set()
        for x in ipa_strings:
            try:
                ipa.add(IpaSequence(x))
            except ValueError:
                pass
        ipa = frozenset(ipa)
        word = Word(lang, form, ipa)
        cls._cache[key] = word
        return word

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()


@dataclass
class AlignedWord:
    """Represent a potentially aligned word."""
    lost_token: Word
    lost_lemma: Word
    known_tokens: Set[Word]  # Translations for the lost token.
    known_lemmas: Set[Word]  # Translations for the lost lemma.
    lost_stems: Set[Stem] = field(default_factory=set)
    known_stems: Set[Stem] = field(default_factory=set)

    def __repr__(self):
        """Only display the most important (form) information."""
        ltf = self.lost_token.form
        llf = self.lost_lemma.form

        def set_str(s: Set[Word]) -> str:
            if s:
                return ','.join([w.form for w in s])
            else:
                return ''

        lkts = set_str(self.known_tokens)
        lkls = set_str(self.known_lemmas)
        return repr(f'{ltf}/{llf}/{lkts}/{lkls}')

    @classmethod
    def from_raw_string(cls, lost_lang: str, known_lang: str, raw_string: str, transcriber: MultilingualTranscriber) -> AlignedWord:
        wf = WordFactory()
        lost_token, lost_lemma, known_tokens, known_lemmas = raw_string.split('|')

        lost_token = wf.get_word(lost_lang, lost_token, transcriber)
        lost_lemma = wf.get_word(lost_lang, lost_lemma, transcriber)
        known_tokens = {
            wf.get_word(known_lang, known_token, transcriber)
            for known_token in known_tokens.split(',')
            if known_token
        }
        known_lemmas = {
            wf.get_word(known_lang, known_lemma, transcriber)
            for known_lemma in known_lemmas.split(',')
            if known_lemma
        }
        return cls(lost_token, lost_lemma, known_tokens, known_lemmas)


@dataclass
class Segment:
    start: int
    end: int
    content: Content
    aligned_contents: Set[Content]
    full_form_start: int = None
    full_form_end: int = None

    def _check_segment_type(self, other: Segment):
        if not isinstance(other, Segment):
            raise TypeError(f'Can only compare with Segment instances.')

    def has_same_span(self, other: Segment) -> bool:
        self._check_segment_type(other)
        return self.start == other.start and self.end == other.end

    def has_reasonable_stem_span(self, other: Segment) -> bool:
        self._check_segment_type(other)
        assert other.full_form_end is not None and other.full_form_start is not None
        # print(repr(self), repr(other))
        return self.start == other.full_form_start and self.end <= other.full_form_end

    def has_same_content(self, other: Segment) -> bool:
        self._check_segment_type(other)
        return bool(self.aligned_contents & other.aligned_contents)

    def has_correct_prediction(self, gold: Segment) -> bool:
        self._check_segment_type(gold)
        if self.has_same_content(gold):
            if g.use_stem:
                return self.has_reasonable_stem_span(gold)
            else:
                return self.has_prefix_span(gold)
        return False

    def has_prefix_span(self, other: Segment) -> bool:
        self._check_segment_type(other)
        return self.start == other.start and self.end <= other.end

    def __str__(self):
        ac_str = ','.join(map(str, self.aligned_contents))
        return f'{self.start}~{self.end}@{self.content}|{ac_str}'


class OverlappingAnnotation(Exception):
    """Annotations have overlapping locations."""


@dataclass
class UnsegmentedSentence(SequenceABC):
    content: Content
    is_lost_ipa: bool
    is_known_ipa: bool
    segmented_content: Content = field(repr=False)
    segments: List[Segment] = field(default_factory=list)
    # annotated: Set[int] = field(default_factory=set, repr=False)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx: int) -> Content:
        return self.content[idx]

    def annotate(self,
                 start: int,
                 end: int,
                 aligned_contents: Union[Content, Set[Content]],
                 full_form_start: Optional[int] = None,
                 full_form_end: Optional[int] = None):
        if isinstance(aligned_contents, (str, IpaSequence)):
            aligned_contents = {aligned_contents}
        segment = Segment(start, end, self.content[start: end + 1], aligned_contents, full_form_start, full_form_end)
        # idx_set = set(range(start, end + 1))
        # if idx_set & self.annotated:
        #     raise OverlappingAnnotation(f'Overlapping locations for {segment}.')
        self.segments.append(segment)  # pylint: disable=no-member
        # self.annotated |= idx_set


@dataclass
class AlignedSentence:
    """Represent a sentence with potentially aligned words."""
    words: List[AlignedWord]

    def __repr__(self):
        """Only display the lost form information."""
        return self.surface_form

    @cached_property
    def surface_form(self) -> str:
        return repr(' '.join([word.lost_token.form for word in self.words]))

    @classmethod
    def from_raw_string(cls, lost_lang: str, known_lang: str, raw_string: str, transcriber: MultilingualTranscriber) -> AlignedSentence:
        words = list()
        for raw_string_per_word in raw_string.split():
            aligned_word = AlignedWord.from_raw_string(lost_lang, known_lang, raw_string_per_word, transcriber)
            words.append(aligned_word)
        return cls(words)

    @property
    def lost_form_length(self):
        return sum([word.lost_token.form_length for word in self.words])

    @property
    def lost_ipa_length(self):

        return sum([word.lost_token.ipa_length for word in self.words])

    @property
    def length(self) -> int:
        if g.input_format == 'ipa':
            return self.lost_ipa_length
        else:
            return self.lost_form_length

    def __getitem__(self, idx: int) -> AlignedWord:
        return self.words[idx]

    def to_unsegmented(self, *,
                       is_lost_ipa: bool = None,
                       is_known_ipa: bool = None,
                       annotated: bool = None) -> UnsegmentedSentence:
        check_explicit_arg(is_lost_ipa, is_known_ipa, annotated)
        if is_lost_ipa:
            warnings.warn('Only one of the ipa sequences is used.')
            content_lst = [str(word.lost_token.main_ipa) for word in self.words]
            content = IpaSequence(''.join(content_lst))
        else:
            content_lst = [word.lost_token.form for word in self.words]
            content = ''.join(content_lst)
        segmented_content = ' '.join(content_lst)
        uss = UnsegmentedSentence(content, is_lost_ipa, is_known_ipa, segmented_content)

        def iter_annotation() -> Iterator[Tuple[int, int, Set[Content], Optional[int], Optional[int]]]:
            offset = 0
            for word in self.words:
                words_or_stems = word.known_stems if g.use_stem else (word.known_tokens | word.known_lemmas)
                full_length = word.lost_token.ipa_length if is_lost_ipa else word.lost_token.form_length
                aligned_contents = {wos.ipa if is_known_ipa else {wos.form} for wos in words_or_stems}
                aligned_contents = union_sets(aligned_contents)
                if aligned_contents:
                    if g.use_stem:
                        for stem in word.lost_stems:
                            yield offset + stem.start, offset + stem.end, aligned_contents, offset, offset + full_length - 1
                    else:
                        yield offset, offset + full_length - 1, aligned_contents, None, None
                offset += full_length

        if annotated:
            offset = 0
            for start, end, aligned_contents, full_form_start, full_form_end in iter_annotation():
                length = end - start + 1
                if length <= g.max_word_length and length >= g.min_word_length:
                    uss.annotate(start, end, aligned_contents,
                                 full_form_start=full_form_start,
                                 full_form_end=full_form_end)
            # for word in self.words:
            #     lwl = word.lost_token.ipa_length if is_lost_ipa else word.lost_token.form_length
            #     if (word.known_tokens or word.known_lemmas) and lwl <= g.max_word_length and lwl >= g.min_word_length:
            #         aligned_contents = set()
            #         for known_word in (word.known_tokens | word.known_lemmas):
            #             if is_known_ipa:
            #                 aligned_contents.update(known_word.ipa)
            #             else:
            #                 aligned_contents.add(known_word.form)
            #         uss.annotate(offset, offset + lwl - 1, aligned_contents)
            #     offset += lwl

        return uss


class NullValue(Exception):
    """Raise this error if a function is applied on a null value in data frames."""


class AlignedCorpus(SequenceABC):
    """Represent a corpus with potentially aligned sentences."""

    def __init__(self, lost_lang: str, known_lang: str, sentences: Sequence[AlignedSentence]):
        self.lost_lang = lost_lang
        self.known_lang = known_lang
        self.sentences = sentences

        # Go through the dataset once to obtain all unique ipa units.
        is_ipa = g.input_format == 'ipa'

        def yield_content(word: Word):
            if is_ipa:
                yield from word.ipa
            else:
                yield from [word.form]

        def gen_lost_word():
            for sentence in sentences:
                for word in sentence.words:
                    # NOTE(j_luo) No need to yield content from lemmas.
                    yield from yield_content(word.lost_token)

        def gen_known_word():
            for sentence in sentences:
                for word in sentence.words:
                    for known_word in word.known_tokens | word.known_lemmas:
                        yield from yield_content(known_word)

        csf = CharSetFactory()
        self.char_sets = {
            lost_lang: csf.get_char_set(gen_lost_word(), lost_lang, is_ipa),
            known_lang: csf.get_char_set(gen_known_word(), known_lang, is_ipa)
        }

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> AlignedSentence:
        return self.sentences[idx]

    @classmethod
    def from_data_path(cls, lost_lang: str, known_lang: str, data_path: Path, transcriber: MultilingualTranscriber) -> AlignedCorpus:
        with data_path.open('r', encoding='utf8') as fin:
            sentences = list()
            for line in fin:
                sentence = AlignedSentence.from_raw_string(lost_lang, known_lang, line.strip(), transcriber)
                sentences.append(sentence)
        return cls(lost_lang, known_lang, sentences)

    @classmethod
    def from_tsv(cls, lost_lang: str, known_lang: str, data_path: str) -> AlignedCorpus:

        def get_word(saved_string: str, strict: bool = True, return_set: bool = False, word_cls=Word):
            if pd.isnull(saved_string):
                if strict:
                    raise NullValue(f'Null value encountered.')
                return set() if return_set else None
            else:
                if return_set:
                    return {word_cls.from_saved_string(s) for s in saved_string.split('|')}
                else:
                    return word_cls.from_saved_string(saved_string)

        df = pd.read_csv(data_path, sep='\t')
        word_info_cols = ['lost_token', 'lost_lemma', 'known_tokens', 'known_lemmas']
        df['lost_token'] = df['lost_token'].apply(get_word)
        df['lost_lemma'] = df['lost_lemma'].apply(get_word, strict=False)
        df['known_tokens'] = df['known_tokens'].apply(get_word, strict=False, return_set=True)
        if g.use_stem:
            # NOTE(j_luo) These two columns are identical if stems are used.
            df['known_lemmas'] = df['known_tokens']
            # Get stems.
            df['lost_stems'] = df['lost_stems'].apply(get_word, strict=False, return_set=True, word_cls=Stem)
            df['known_stems'] = df['known_stems'].apply(get_word, strict=False, return_set=True, word_cls=Stem)
            word_info_cols.extend(['lost_stems', 'known_stems'])
        else:
            df['known_lemmas'] = df['known_lemmas'].apply(get_word, strict=False, return_set=True)
        df['aligned_word'] = df[word_info_cols].apply(lambda item: AlignedWord(*item), axis=1)

        sentences = list()
        for sentence_idx, group in df.groupby('sentence_idx', sort=True)['word_idx', 'aligned_word']:
            words = list(group.sort_values('word_idx')['aligned_word'])
            sentences.append(AlignedSentence(words))

        return cls(lost_lang, known_lang, sentences)

    def to_df(self) -> pd.DataFrame:
        if g.use_stem:
            raise NotImplementedError(f'Saving stems are not supported right now.')

        data = list()
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx, word in enumerate(sentence.words):
                lt = word.lost_token
                ll = word.lost_lemma
                kts = word.known_tokens
                kls = word.known_lemmas
                data.append((s_idx, w_idx, str(lt), str(ll), '|'.join(map(str, kts)), '|'.join(map(str, kls))))
        df = pd.DataFrame(data,
                          columns=['sentence_idx', 'word_idx',
                                   'lost_token', 'lost_lemma',
                                   'known_tokens', 'known_lemmas'])
        return df

    def to_tsv(self, data_path: str):
        df = self.to_df()
        df.to_csv(data_path, sep='\t', index=None)
