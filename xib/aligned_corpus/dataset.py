from __future__ import annotations

from torch.utils.data import Dataset

from dev_misc import g
from xib.aligned_corpus.corpus import AlignedCorpus
from dataclasses import dataclass


@dataclass
class AlignedDatasetItem:
    ...  # FIXME(j_luo) fill in this


class AlignedDataset(Dataset):
    """A subclass of Dataset that deals with AlignedCorpus."""

    def __init__(self, corpus: AlignedCorpus):
        ...  # FIXME(j_luo) fill in this. Use g.max_segment_length.

    def __len__(self):
        ...  # FIXME(j_luo) fill in this

    def __getitem__(self, idx: int) -> AlignedDatasetItem:
        ...  # FIXME(j_luo) fill in this
