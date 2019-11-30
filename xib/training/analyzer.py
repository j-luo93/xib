"""
Analyzers package methods that are shared between evaluators and trainers.
"""

from typing import Dict

import torch

from dev_misc import FT, g
from dev_misc.devlib import get_length_mask
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics
from xib.data_loader import ContinuousTextIpaBatch
from xib.ipa import should_include
from xib.model.decipher_model import DecipherModel, DecipherModelReturn
from xib.model.lm_model import Cat


class LMAnalyzer:

    def analyze(self, scores: Dict[Cat, FT]) -> Metrics:
        metrics = Metrics()
        total_loss = 0.0
        total_weight = 0.0
        for name, (losses, weights) in scores.items():
            if should_include(g.feat_groups, name):
                loss = (losses * weights).sum()
                weight = weights.sum()
                total_loss += loss
                total_weight += weight
                loss = Metric(f'loss_{name.snake}', loss, weight)
                metrics += loss
        metrics += Metric('loss', total_loss, total_weight)
        return metrics


class DecipherAnalyzer:

    def analyze(self, ret: DecipherModelReturn, batch: ContinuousTextIpaBatch) -> Metrics:
        metrics = Metrics()

        is_unique = ret.packed_words.is_unique
        modified_log_probs = ret.probs.sample_log_probs * g.concentration + (~is_unique).float() * (-999.9)
        sample_scores = ret.scores.lm_score + ret.scores.readable_score + ret.scores.unreadable_score
        sample_probs = modified_log_probs.log_softmax(dim='sample').exp()
        utility = (sample_probs * sample_scores).sum()
        total_loss = Metric('total_loss', -utility, batch.batch_size)
        metrics += total_loss

        return metrics
