from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from dev_misc import LT, g
from xib.aligned_corpus.corpus import AlignedCorpus, AlignedSentence


def split_by_max_length(lengths: Sequence[int], max_length: int) -> List[Tuple[int, int]]:
    ret = list()
    cum_lengths = [0]
    for length in lengths:
        cum_lengths.append(cum_lengths[-1] + length)
    start = 0
    end = 1
    while end < len(cum_lengths):
        if cum_lengths[end] - cum_lengths[start] <= max_length:
            end += 1
            continue

        if end > start + 1:
            ret.append((start, end - 1))
            start = end - 1
        else:
            start = end
        end = start + 1

    if end > start + 1:
        ret.append((start, end - 1))

    return ret


@dataclass
class AlignedDatasetItem:
    length: int
    feat_matrix: LT


class AlignedDataset(Dataset):
    """A subclass of Dataset that deals with AlignedCorpus."""

    def __init__(self, corpus: AlignedCorpus):
        self.corpus = corpus
        self.data = list()
        for sentence in self.corpus.sentences:
            word_lengths = map(len, sentence.words)
            splits = split_by_max_length(word_lengths, g.max_segment_length)
            for start, end in splits:
                truncated_sentence = AlignedSentence(sentence.words[start: end])
                self.data.append(truncated_sentence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> AlignedDatasetItem:
        sentence = self.data[idx]
        length = len(sentence)
        feat_matrix = torch.stack(
            [word.lost_word.main_ipa.feat_matrix for word in sentence.words],
            dim=0
        )
        return AlignedDatasetItem(length, feat_matrix)
