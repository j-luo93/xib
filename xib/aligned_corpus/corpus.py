from __future__ import annotations

import warnings
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import (IO, ClassVar, Dict, FrozenSet, Iterable, List, Optional,
                    Sequence, Set, Tuple, TypeVar, Union)

import numpy as np
import pandas as pd
import torch

from dev_misc import add_argument, g
from dev_misc.devlib import BaseBatch, batch_class, get_array, get_length_mask
from dev_misc.devlib.named_tensor import NoName, Rename
from dev_misc.utils import Singleton, check_explicit_arg
from xib.aligned_corpus.ipa_sequence import IpaSequence
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


@dataclass
class AlignedWord:
    """Represent a potentially aligned word."""
    lost_token: Word
    lost_lemma: Word
    known_tokens: Set[Word]  # Translations for the lost token.
    known_lemmas: Set[Word]  # Translations for the lost lemma.

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


Content = TypeVar('Content', str, IpaSequence)


@dataclass
class Segment:
    start: int
    end: int
    content: Content
    aligned_contents: Set[Content]

    def is_same_span(self, other: Segment) -> bool:
        if not isinstance(other, Segment):
            raise TypeError(f'Can only compare with Segment instances.')
        return self.start == other.start and self.end == other.end

    def is_prefix_span(self, other: Segment) -> bool:
        if not isinstance(other, Segment):
            raise TypeError(f'Can only compare with Segment instances.')
        return self.start == other.start and self.end <= other.end

    def __str__(self):
        ac_str = ','.join(map(str, self.aligned_contents))
        return f'{self.start}~{self.end}@{self.content}|{ac_str}'


class OverlappingAnnotation(Exception):
    """Annotations have overlapping locations."""


@dataclass
class UnsegmentedSentence(SequenceABC):
    content: Content
    is_ipa: bool
    segmented_content: Content = field(repr=False)
    segments: List[Segment] = field(default_factory=list)
    annotated: Set[int] = field(default_factory=set, repr=False)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx: int) -> Content:
        return self.content[idx]

    def annotate(self, start: int, end: int, aligned_contents: Union[Content, Set[Content]]):
        if isinstance(aligned_contents, (str, IpaSequence)):
            aligned_contents = {aligned_contents}
        segment = Segment(start, end, self.content[start: end + 1], aligned_contents)
        idx_set = set(range(start, end + 1))
        if idx_set & self.annotated:
            raise OverlappingAnnotation(f'Overlapping locations for {segment}.')
        self.segments.append(segment)  # pylint: disable=no-member
        self.annotated |= idx_set


@dataclass
class AlignedSentence:
    """Represent a sentence with potentially aligned words."""
    words: List[AlignedWord]

    def __repr__(self):
        """Only display the lost form information."""
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

    def __getitem__(self, idx: int) -> AlignedWord:
        return self.words[idx]

    def to_unsegmented(self, *,
                       is_ipa: bool = None,
                       annotated: bool = None) -> UnsegmentedSentence:
        check_explicit_arg(is_ipa, annotated)
        if is_ipa:
            warnings.warn('Only one of the ipa sequences is used.')
            content_lst = [str(list(word.lost_token.ipa)[0]) for word in self.words]
            content = IpaSequence(''.join(content_lst))
        else:
            content_lst = [word.lost_token.form for word in self.words]
            content = ''.join(content_lst)
        segmented_content = ' '.join(content_lst)
        uss = UnsegmentedSentence(content, is_ipa, segmented_content)
        if annotated:
            offset = 0
            for word in self.words:
                lwl = word.lost_token.ipa_length if is_ipa else word.lost_token.form_length
                if (word.known_tokens or word.known_lemmas) and lwl <= g.max_word_length and lwl >= g.min_word_length:
                    aligned_contents = set()
                    for known_word in (word.known_tokens | word.known_lemmas):
                        if is_ipa:
                            aligned_contents.update(known_word.ipa)
                        else:
                            aligned_contents.add(known_word.form)
                    uss.annotate(offset, offset + lwl - 1, aligned_contents)
                offset += lwl

        return uss


def _gather_units(iterable: Iterable[Content]) -> Tuple[List[Content], Dict[Content, int]]:
    all_units = set()
    for content in iterable:
        try:
            all_units.update(content.cv_list)
        except:
            all_units.update(content)
    id2unit = sorted(all_units, key=str)
    unit2id = {u: i for i, u in enumerate(id2unit)}
    return id2unit, unit2id


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
                    for lost_word in [word.lost_token, word.lost_lemma]:
                        yield from yield_content(lost_word)

        def gen_known_word():
            for sentence in sentences:
                for word in sentence.words:
                    for known_word in word.known_tokens | word.known_lemmas:
                        yield from yield_content(known_word)

        lost_id2unit, lost_unit2id = _gather_units(gen_lost_word())
        known_id2unit, known_unit2id = _gather_units(gen_known_word())

        self.id2unit = {lost_lang: lost_id2unit, known_lang: known_id2unit}
        self.unit2id = {lost_lang: lost_unit2id, known_lang: known_unit2id}

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
                    try:
                        ret.add(Word.from_saved_string(s))
                    except ValueError:
                        pass
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


class Vocabulary:

    add_argument('add_infinitive', dtype=bool, default=False)

    def __init__(self):

        def has_proper_length(content: Content) -> bool:
            l = len(content)
            return g.min_word_length <= l <= g.max_word_length

        def gen_word(io: IO):
            for line in io:
                try:
                    word = Word.from_saved_string(line.strip())
                    yield from word.ipa
                except ValueError:
                    pass

        def expand(vocab: Iterable[Content]):
            if g.add_infinitive:
                for content in vocab:
                    if str(content).endswith('ən'):
                        yield from [content, content[:-2]]
                    else:
                        yield content
            else:
                return vocab

        with open(g.vocab_path, 'r', encoding='utf8') as fin:
            self.vocab = get_array(sorted(set(expand(filter(has_proper_length, gen_word(fin)))),
                                          key=str))

            self.vocab_length = torch.LongTensor(list(map(len, self.vocab)))
            max_len = self.vocab_length.max().item()
            self.vocab_source_padding = ~get_length_mask(self.vocab_length, max_len)
            self.vocab_length.rename_('vocab')
            self.vocab_source_padding.rename_('vocab', 'length')

            feat_matrix = [word.feat_matrix for word in self.vocab]
            self.vocab_feat_matrix = torch.nn.utils.rnn.pad_sequence(feat_matrix, batch_first=True)
            self.vocab_feat_matrix.rename_('vocab', 'length', 'feat_group')

            with Rename(self.vocab_feat_matrix, vocab='batch'):
                vocab_dense_feat_matrix = convert_to_dense(self.vocab_feat_matrix)
            self.vocab_dense_feat_matrix = {k: v.rename(batch='vocab') for k, v in vocab_dense_feat_matrix.items()}

            # Get the entire set of units from vocab.
            self.id2unit, self.unit2id = _gather_units(self.vocab)

            # Now indexify the vocab. Gather feature matrices for units as well.
            indexed_segments = np.zeros([len(self.vocab), max_len], dtype='int64')
            unit_feat_matrix = dict()
            for i, segment in enumerate(self.vocab):
                indexed_segments[i, range(len(segment))] = [self.unit2id[u] for u in segment.cv_list]
                for j, u in enumerate(segment.cv_list):
                    if u not in unit_feat_matrix:
                        unit_feat_matrix[u] = segment.feat_matrix[j]
            unit_feat_matrix = [unit_feat_matrix[u] for u in self.id2unit]
            unit_feat_matrix = torch.nn.utils.rnn.pad_sequence(unit_feat_matrix, batch_first=True)
            self.unit_feat_matrix = unit_feat_matrix.unsqueeze(dim=1)
            self.indexed_segments = torch.from_numpy(indexed_segments)

            # Use dummy length to avoid the trouble later on.
            # HACK(j_luo) Have to provide 'length'.
            self.unit_feat_matrix.rename_('unit', 'length', 'feat_group')
            self.indexed_segments.rename_('vocab', 'length')
            with Rename(self.unit_feat_matrix, unit='batch'):
                unit_dense_feat_matrix = convert_to_dense(self.unit_feat_matrix)
            self.unit_dense_feat_matrix = {
                k: v.rename(batch='unit')
                for k, v in unit_dense_feat_matrix.items()
            }

    def __len__(self):
        return len(self.vocab)
