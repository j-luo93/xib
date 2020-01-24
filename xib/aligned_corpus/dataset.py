from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from dev_misc import LT, add_argument, g
from xib.aligned_corpus.corpus import AlignedCorpus, AlignedSentence


def split_by_length(lengths: Sequence[int], max_length: int, min_length: int) -> List[Tuple[int, int]]:
    ret = list()
    cum_lengths = [0]
    for length in lengths:
        cum_lengths.append(cum_lengths[-1] + length)

    start = 0
    while start < len(lengths):
        end = start + 1
        while end < len(cum_lengths) - 1 and cum_lengths[end + 1] <= max_length:
            end += 1
        if min_length <= cum_lengths[end] - cum_lengths[start] <= max_length:
            ret.append((start, end))
        start = end
    return ret

# def split_by_length(lengths: Sequence[int], max_length: int, min_length: int) -> List[Tuple[int, int]]:
#     ret = list()
#     cum_lengths = [0]
#     for length in lengths:
#         cum_lengths.append(cum_lengths[-1] + length)
#     start = 0
#     end = 1
#     while end < len(cum_lengths):
#         seg_length = cum_lengths[end] - cum_lengths[start]
#         if seg_length <= max_length:
#             end += 1
#             continue

#         if end > start + 1 and seg_length >= min_length:
#             ret.append((start, end - 1))
#             start = end - 1
#         else:
#             start = end
#         end = start + 1

#     if end > start + 1 and seg_length >= min_length:
#         ret.append((start, end - 1))

#     return ret


@dataclass
class AlignedDatasetItem:
    sentence: AlignedSentence
    length: int
    feat_matrix: LT


class AlignedDataset(Dataset):
    """A subclass of Dataset that deals with AlignedCorpus."""

    add_argument('noiseless', dtype=bool, default=False)
    add_argument('freq_hack', dtype=bool, default=False)

    def __init__(self, corpus: AlignedCorpus):
        self.corpus = corpus
        self.data = list()
        # HACK(j_luo)
        if g.freq_hack:
            _data_str = set()
        for sentence in self.corpus.sentences:
            word_lengths = [word.lost_token.form_length for word in sentence.words]
            splits = split_by_length(word_lengths, g.max_segment_length, g.min_word_length)
            for start, end in splits:
                truncated_sentence = AlignedSentence(sentence.words[start: end])

                to_add = False
                if g.noiseless:
                    uss = truncated_sentence.to_unsegmented(is_known_ipa=True,
                                                            is_lost_ipa=g.input_format == 'ipa',
                                                            annotated=True)
                    if uss.segments and any(g.min_word_length <= segment.end - segment.start <= g.max_word_length for segment in uss.segments):
                        to_add = True
                else:
                    to_add = True

                if to_add:
                    if g.freq_hack and str(truncated_sentence) in _data_str:
                        continue
                    self.data.append(truncated_sentence)
                    if g.freq_hack:
                        _data_str.add(str(truncated_sentence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> AlignedDatasetItem:
        sentence = self.data[idx]
        length = sentence.length
        feat_matrix = torch.cat(
            [word.lost_token.main_ipa.feat_matrix for word in sentence.words],
            dim=0
        )
        return AlignedDatasetItem(sentence, length, feat_matrix)
