from __future__ import annotations

import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import (ClassVar, Dict, FrozenSet, List, Optional, Sequence, Set,
                    Tuple, TypeVar, Union)

import pandas as pd

from dev_misc.devlib import BaseBatch, batch_class
from dev_misc.utils import Singleton, check_explicit_arg
from xib.aligned_corpus.ipa_sequence import IpaSequence
from xib.aligned_corpus.transcriber import MultilingualTranscriber


@batch_class(eq=True, frozen=True)
class Word(BaseBatch):
    """A word with form and ipa."""
    lang: str
    form: str
    ipa: FrozenSet[IpaSequence]

    def __len__(self):
        return len(self.form)

    def __str__(self):
        ipa_str = ','.join(map(str, self.ipa))
        return f'{self.lang};{self.form};{ipa_str}'

    @classmethod
    def from_saved_string(cls, saved_string: str) -> Word:
        lang, form, ipa_str = saved_string.split(';')
        ipa = frozenset({IpaSequence(s) for s in ipa_str.split(',')})
        return cls(lang, form, ipa)

    @property
    def main_ipa(self) -> IpaSequence:
        # HACK(j_luo) Only take one of them.
        return list(self.ipa)[0]


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


@batch_class
class AlignedWord(BaseBatch):
    """Represent a potentially aligned word."""
    lost_token: Word
    lost_lemma: Word
    known_tokens: Set[Word]  # Translations for the lost token.
    known_lemmas: Set[Word]  # Translations for the lost lemma.

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


Content = TypeVar('Content', str, IpaSequence)


@dataclass
class Segment:
    start: int
    end: int
    content: Content
    aligned_contents: Set[Content]

    def is_same_span(self, other: Segment) -> bool:
        return self.start == other.start and self.end == other.end

    def __str__(self):
        return f'{self.start}~{self.end}@{self.content}|{self.aligned_contents}'


class OverlappingAnnotation(Exception):
    """Annotations have overlapping locations."""


@dataclass
class UnsegmentedSentence(SequenceABC):
    content: Content
    is_ipa: bool
    segments: List[Segment] = field(default_factory=list)
    annotated: Set[int] = field(default_factory=set, repr=False)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx: int) -> Content:
        return self.content[idx]

    def annotate(self, start: int, end: int, aligned_contents: Content):
        segment = Segment(start, end, self.content[start: end + 1], aligned_contents)
        idx_set = set(range(start, end + 1))
        if idx_set & self.annotated:
            raise OverlappingAnnotation(f'Overlapping locations for {segment}.')
        self.segments.append(segment)  # pylint: disable=no-member
        self.annotated |= idx_set


@batch_class
class AlignedSentence(SequenceABC, BaseBatch):
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
            content = [str(list(word.lost_token.ipa)[0]) for word in self.words]
        else:
            content = [word.lost_token.form for word in self.words]
        content = ''.join(content)
        uss = UnsegmentedSentence(content, is_ipa)
        if annotated:
            offset = 0
            for word in self.words:
                lwl = len(word.lost_token)
                if word.known_tokens or word.known_lemmas:
                    aligned_contents = {
                        word.ipa if is_ipa else word.form
                        for word in word.known_tokens | word.known_lemmas
                    }
                    uss.annotate(offset, offset + lwl - 1, aligned_contents)
                offset += len(word.lost_token)

        return uss


class AlignedCorpus(SequenceABC):
    """Represent a corpus with potentially aligned sentences."""

    def __init__(self, lost_lang: str, known_lang: str, sentences: Sequence[AlignedSentence]):
        self.lost_lang = lost_lang
        self.known_lang = known_lang
        self.sentences = sentences

        # Go through the dataset once to obtain all unique ipa units.
        ipa_units = {lost_lang: set(), known_lang: set()}
        for sentence in self.sentences:
            for word in sentence.words:
                ipa_units[lost_lang].update(word.lost_token.main_ipa.cv_list)
                ipa_units[lost_lang].update(word.lost_lemma.main_ipa.cv_list)
                for known_word in word.known_tokens | word.known_lemmas:
                    ipa_units[known_lang].update(known_word.main_ipa.cv_list)
        self.id2unit = {
            lang: sorted(ipa_units[lang], key=str)
            for lang in [lost_lang, known_lang]
        }
        self.unit2id = {
            lang: {
                u: i
                for i, u in enumerate(self.id2unit[lang])
            }
            for lang in [lost_lang, known_lang]
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
        df = pd.read_csv(data_path, sep='\t')
        df['lost_token'] = df['lost_token'].apply(Word.from_saved_string)
        df['lost_lemma'] = df['lost_lemma'].apply(Word.from_saved_string)

        def get_set_of_words(saved_string: str) -> Set[Word]:
            ret = set()
            try:
                for s in saved_string.split('|'):
                    if s:
                        ret.add(Word.from_saved_string(s))
            except AttributeError:
                pass
            return ret

        df['known_tokens'] = df['known_tokens'].apply(get_set_of_words)
        df['known_lemmas'] = df['known_lemmas'].apply(get_set_of_words)
        df['aligned_word'] = df[['lost_token', 'lost_lemma', 'known_tokens',
                                 'known_lemmas']].apply(lambda item: AlignedWord(*item), axis=1)

        sentences = list()
        for sentence_idx, group in df.groupby('sentence_idx', sort=True)['word_idx', 'aligned_word']:
            words = list(group.sort_values('word_idx')['aligned_word'])
            sentences.append(AlignedSentence(words))

        return cls(lost_lang, known_lang, sentences)

    def to_df(self) -> pd.DataFrame:

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
