from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Dict

import torch

from dev_misc import FT, LT, get_tensor
from dev_misc.devlib import BaseBatch, batch_class
from dev_misc.trainlib import Metric, Metrics, Tracker
from xib.data_loader import ContinuousTextDataLoader, ContinuousTextIpaBatch
from xib.ipa.process import B, I, O
from xib.model.decipher_model import DecipherModel, PackedWords


@batch_class
class SearchResult(BaseBatch):
    scores: Dict


class BaseSearcher(ABC):

    def __init__(self, model: DecipherModel, dl: ContinuousTextDataLoader):
        self.model = model
        self.dl = dl

    def evaluate(self) -> Metrics:
        metrics = Metrics()
        with torch.no_grad():
            self.model.eval()
            for batch in self.dl:
                search_result = self.search(batch)
        return metrics

    @abstractmethod
    def search(self, batch: ContinuousTextIpaBatch) -> SearchResult: ...


class BruteForceSearcher(BaseSearcher):

    def search(self, batch: ContinuousTextIpaBatch) -> SearchResult:
        max_length = batch.lengths.max().item()
        samples = get_tensor(torch.LongTensor(list(product([B, I, O], repeat=max_length))))
        samples.rename_('sample', 'length')
        samples = samples.align_to('batch', 'sample', 'length').expand(batch.batch_size, -1, -1)
        scores = self.model.get_scores(samples, batch)
        return scores
        # return SearchResult(
        #     scores
        # )


class BeamSearcher(BaseSearcher):
    ...  # FIXME(j_luo) fill in this
