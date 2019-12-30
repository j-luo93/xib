from typing import List

from dev_misc.devlib import BaseBatch, batch_class
from dev_misc.trainlib.base_data_loader import BaseDataLoader
from xib.aligned_corpus.dataset import AlignedDatasetItem


@batch_class
class BaseBatch:
    ...  # FIXME(j_luo) fill in this


def collate_aligned_dataset_items(items: List[AlignedDatasetItem]):
    ...  # FIXME(j_luo) fill in this


class AlignedDataLoader(BaseDataLoader):

    collate_fn = ...  # FIXME(j_luo) fill in this

    def __iter__(self):
        for batch in super().__iter__():
            return batch.cuda()
