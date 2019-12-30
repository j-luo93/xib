from __future__ import annotations

import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import (ClassVar, Dict, List, Optional, Sequence, Set, Tuple,
                    TypeVar, Union)

import pandas as pd

from dev_misc.utils import Singleton, check_explicit_arg
from xib.aligned_corpus.ipa_sequence import IpaSequence
from xib.aligned_corpus.transcriber import MultilingualTranscriber


@dataclass
class Word:
    """A word with form and ipa."""
    lang: str
    form: str
    ipa: Set[IpaSequence]

    def __len__(self):
        return len(self.form)


_Signature = Tuple[str, str]


class WordFactory(Singleton):

    _cache: ClassVar[Dict[_Signature, Word]] = dict()

    def get_word(self, lang: str, form: str, transcriber: MultilingualTranscriber) -> Word:
        cls = type(self)
        key = (lang, form)
        if key in cls._cache:
            return cls._cache[key]

        ipa = transcriber.transcribe(form, lang)
        ipa = {IpaSequence(str(x)) for x in ipa}
        word = Word(lang, form, ipa)
        cls._cache[key] = word
        return word

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()


@dataclass
class AlignedWord:
    """Represent a potentially aligned word."""
    lost_word: Word
    known_word: Optional[Word] = None

    @classmethod
    def from_raw_string(cls, lost_lang: str, known_lang: str, raw_string: str, transcriber: MultilingualTranscriber) -> AlignedWord:
        wf = WordFactory()
        lost_form, *known_form = raw_string.split('|')
        if len(known_form) > 1:
            raise ValueError(f'Raw string {raw_string} has more than one "|".')

        lost_word = wf.get_word(lost_lang, lost_form, transcriber)
        known_word = None
        if known_form:
            known_word = wf.get_word(known_lang, known_form[0], transcriber)
        return cls(lost_word, known_word=known_word)


Text = TypeVar('Text', str, IpaSequence)


@dataclass
class Segment:
    start: int
    end: int
    aligned_text: Text


class OverlappingAnnotation(Exception):
    """Annotations have overlapping locations."""


@dataclass
class UnsegmentedSentence(SequenceABC):
    text: Text
    is_ipa: bool
    segments: List[Segment] = field(default_factory=list)
    annotated: Set[int] = field(default_factory=set, repr=False)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx: int) -> Text:
        return self.text[idx]

    def annotate(self, start: int, end: int, aligned_text: Text):
        segment = Segment(start, end, aligned_text)
        idx_set = set(range(start, end + 1))
        if idx_set & self.annotated:
            raise OverlappingAnnotation(f'Overlapping locations for {segment}.')
        self.segments.append(segment)  # pylint: disable=no-member
        self.annotated |= idx_set


@dataclass
class AlignedSentence(SequenceABC):
    """Represent a sentence with potentially aligned words."""
    words: List[AlignedWord]

    @classmethod
    def from_raw_string(cls, lost_lang: str, known_lang: str, raw_string: str, transcriber: MultilingualTranscriber) -> AlignedSentence:
        words = list()
        for raw_string_per_word in raw_string.split():
            aligned_word = AlignedWord.from_raw_string(lost_lang, known_lang, raw_string_per_word, transcriber)
            words.append(aligned_word)
        return cls(words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx: int) -> AlignedWord:
        return self.words[idx]

    def to_unsegmented(self, *, is_ipa: bool = None, annotated: bool = None) -> UnsegmentedSentence:
        check_explicit_arg(is_ipa, annotated)
        if is_ipa:
            warnings.warn('Only one of the ipa sequences is used.')
            text = [str(list(word.lost_word.ipa)[0]) for word in self.words]
        else:
            text = [word.lost_word.form for word in self.words]
        text = ''.join(text)
        uss = UnsegmentedSentence(text, is_ipa)
        if annotated:
            offset = 0
            for word in self.words:
                lwl = len(word.lost_word)
                if word.known_word is not None:
                    aligned_text = word.known_word.ipa if is_ipa else word.known_word.form
                    uss.annotate(offset, offset + lwl - 1, aligned_text)
                offset += len(word.lost_word)

        return uss


class AlignedCorpus(SequenceABC):
    """Represent a corpus with potentially aligned sentences."""

    def __init__(self, sentences: Sequence[AlignedSentence]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int) -> AlignedSentence:
        return self.sentences[idx]

    @classmethod
    def from_tsv(self, tsv_path: str) -> AlignedCorpus:
        df = pd.read_csv(tsv_path, sep='\t')

        def load_ipa(save_ipa_str: Union[None, str]) -> Union[None, Set[IpaSequence]]:
            if save_ipa_str is None:
                return None
            ipas = save_ipa_str.split('|')
            ret = set()
            for ipa in ipas:
                ipa = IpaSequence.from_saved_string(ipa)
                ret.add(ipa)
            return ret

        df['lost_ipa'] = df['lost_ipa'].apply(load_ipa)
        df['known_ipa'] = df['known_ipa'].apply(load_ipa)

        def get_word(item: Tuple[Union[None, str], Union[None, str], Union[None, Set[IpaSequence]]]) -> Union[None, Word]:
            lang, form, ipa = item
            if lang is None or form is None or ipa is None:
                return None
            return Word(lang, form, ipa)

        df['lost_word'] = df[['lost_lang', 'lost_form', 'lost_ipa']].apply(get_word, axis=1)
        df['known_word'] = df[['known_lang', 'known_form', 'known_ipa']].apply(get_word, axis=1)
        df['aligned_word'] = df[['lost_word', 'known_word']].apply(lambda item: AlignedWord(*item), axis=1)

        sentences = list()
        for sentence_idx, group in df.groupby('sentence_idx', sort=True)['word_idx', 'aligned_word']:
            words = list(group.sort_values('word_idx')['aligned_word'])
            sentences.append(AlignedSentence(words))

        return AlignedCorpus(sentences)

    def to_df(self) -> pd.DataFrame:

        def safe_get_word_info(word: Union[None, Word]):
            try:
                lang = word.lang
                form = word.form
                save_ipa_str = '|'.join(x.save() for x in word.ipa)
                return lang, form, save_ipa_str
            except AttributeError:
                return None, None, None

        data = list()
        for s_idx, sentence in enumerate(self.sentences):
            for w_idx, word in enumerate(sentence.words):
                lw = word.lost_word
                kw = word.known_word
                lw_info = safe_get_word_info(lw)
                kw_info = safe_get_word_info(kw)
                data.append((s_idx, w_idx) + lw_info + kw_info)
        df = pd.DataFrame(data,
                          columns=['sentence_idx', 'word_idx', 'lost_lang',
                                   'lost_form', 'lost_ipa', 'known_lang',
                                   'known_form', 'known_ipa'])
        return df

    def to_tsv(self, tsv_path: str):
        df = self.to_df()
        df.to_csv(tsv_path, sep='\t', index=None)
