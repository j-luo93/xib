from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from typing import Dict, Tuple

import torch

from dev_misc import FT, LT, get_tensor
from dev_misc.devlib import BaseBatch, batch_class
from dev_misc.trainlib import Metric, Metrics, Tracker
from xib.data_loader import ContinuousTextDataLoader, ContinuousTextIpaBatch
from xib.ipa.process import B, I, O


class BaseSearcher(ABC):

    @abstractmethod
    def search(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]: ...


class BruteForceSearcher(BaseSearcher):

    def search(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]:
        max_length = lengths.max().item()
        samples = get_tensor(torch.LongTensor(list(product([B, I, O], repeat=max_length))))
        samples.rename_('sample', 'length')
        bs = label_log_probs.size('batch')
        samples = samples.align_to('batch', 'sample', 'length').expand(bs, -1, -1)
        sample_log_probs = label_log_probs.gather('label', samples)
        return samples, sample_log_probs


class BeamSearcher(BaseSearcher):
    ...  # FIXME(j_luo) fill in this
