from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from dev_misc import LT, add_argument, g
from dev_misc.utils import Singleton
from xib.aligned_corpus.corpus import AlignedCorpus, AlignedSentence

# def split_by_length(lengths: Sequence[int], max_length: int, min_length: int) -> List[Tuple[int, int]]:
#     ret = list()
#     cum_lengths = [0]
#     for length in lengths:
#         cum_lengths.append(cum_lengths[-1] + length)

#     start = 0
#     while start < len(lengths):
#         end = start + 1
#         while end < len(cum_lengths) - 1 and cum_lengths[end + 1] <= max_length:
#             end += 1
#         if min_length <= cum_lengths[end] - cum_lengths[start] <= max_length:
#             ret.append((start, end))
#         start = end
#     return ret


@dataclass
class AlignedDatasetItem:
    sentence: AlignedSentence
    length: int
    feature_matrix: LT


class AlignedDataset:
    """A subclass of Dataset that deals with AlignedCorpus."""

    add_argument('noiseless', dtype=bool, default=False)
    add_argument('freq_hack', dtype=bool, default=False)
    add_argument('min_segment_length', dtype=int)
    add_argument('max_segment_length', default=100, dtype=int)

    def __init__(self, corpus: AlignedCorpus):
        self.corpus = corpus
        self.data = list()
        # HACK(j_luo)
        if g.freq_hack:
            logging.warning('FREQ HACK is used.')
            _data_str = set()
        min_length = g.min_word_length if g.min_segment_length is None else g.min_segment_length
        for sentence in self.corpus.sentences:
            word_lengths = [word.lost_token.form_length for word in sentence.words]
            to_add = g.min_word_length <= sum(word_lengths) <= g.max_segment_length
            # splits = split_by_length(word_lengths, 100, min_length)
            # splits = [(0, sum(word_lengths) + 1)]
            # for start, end in splits:
            #     truncated_sentence = AlignedSentence(sentence.words[start: end])

            #     to_add = False
            #     if g.noiseless:
            #         uss = truncated_sentence.to_unsegmented(is_known_ipa=True,
            #                                                 is_lost_ipa=g.input_format == 'ipa',
            #                                                 annotated=True)
            #         if uss.segments:
            #             for segment in uss.segments:
            #                 if any(g.min_word_length <= ss.end - ss.start + 1 <= g.max_word_length for ss in segment.single_segments):
            #                     to_add = True
            #                     break
            #     else:
            #         to_add = True
            if to_add:
                if g.freq_hack and str(sentence) in _data_str:
                    continue
                self.data.append(sentence)
                if g.freq_hack:
                    _data_str.add(str(sentence))
        total_char = sum([sentence.length for sentence in self.data])
        logging.info(f'There are {total_char} characters in total.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> AlignedDatasetItem:
        sentence = self.data[idx]
        length = sentence.length
        feature_matrix = torch.cat(
            [word.lost_token.main_ipa.feature_matrix for word in sentence.words],
            dim=0
        )
        return AlignedDatasetItem(sentence, length, feature_matrix)


class AlignedDatasetFactory(Singleton):

    _instances: ClassVar[str, AlignedDataset] = dict()

    def get_dataset(self, lost_lang: str, known_lang: str, data_path):
        cls = type(self)
        data_path = str(data_path)
        if data_path not in cls._instances:
            corpus = AlignedCorpus.from_tsv(lost_lang, known_lang, data_path)
            dataset = AlignedDataset(corpus)
            cls._instances[data_path] = dataset
        return cls._instances[data_path]
