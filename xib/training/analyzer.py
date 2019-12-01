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


def _compute_utility(logits: FT, sample_scores: FT) -> FT:
    sample_log_probs = logits.log_softmax(dim='sample')
    utility = (sample_log_probs.exp() * sample_scores).sum()
    return utility


class DecipherAnalyzer:

    def analyze(self, model_ret: DecipherModelReturn, batch: ContinuousTextIpaBatch) -> Metrics:
        if g.supervised:
            return self._analyze_supervised(model_ret, batch)
        else:
            return self._analyze_unsupervised(model_ret, batch)

    def _analyze_supervised(self, model_ret: DecipherModelReturn, batch: ContinuousTextIpaBatch) -> Metrics:
        metrics = Metrics()
        if g.train_phi:
            sample_scores = model_ret.scores.phi_score
            sample_scores = sample_scores.align_to('batch', 'sample')
            gold_log_probs = sample_scores.log_softmax(dim='sample')[:, 0]
            total_loss = Metric('total_loss', -gold_log_probs.sum(), batch.batch_size)
        else:
            target_log_probs = model_ret.probs.label_log_probs.gather('label', batch.gold_tag_seqs)
            weight = (~batch.source_padding).float().align_as(target_log_probs)
            total_loss = (target_log_probs * weight).sum()
            total_loss = Metric('total_loss', -total_loss, weight.sum())
        metrics += total_loss
        return metrics

    def _analyze_unsupervised(self, model_ret: DecipherModelReturn, batch: ContinuousTextIpaBatch) -> Metrics:
        metrics = Metrics()
        # TODO(j_luo) Check the sample scores for hyps that are dummies (i.e., the length of the segment is too small to get beam_size hyps).
        is_unique = model_ret.packed_words.is_unique
        modified_logits = model_ret.probs.sample_log_probs * g.concentration + (~is_unique).float() * (-999.9)
        sample_scores = model_ret.scores.phi_score
        ptb_sample_scores = model_ret.ptb_scores.phi_score
        duplicates = model_ret.duplicates
        with NoName(ptb_sample_scores):
            ptb_sample_scores[duplicates] = -999.9
        bs = sample_scores.size('batch')
        ptb_sample_scores = ptb_sample_scores.unflatten('batch', [('batch', bs), ('contrast', g.n_times * 2)])
        sample_scores = sample_scores.align_as(ptb_sample_scores)
        all_scores = torch.cat([sample_scores, ptb_sample_scores], dim='contrast')
        all_probs = all_scores.log_softmax(dim='contrast').exp()
        sample_probs = all_probs.align_to(..., 'contrast')[..., 0]
        utility = _compute_utility(modified_logits, sample_probs)
        total_loss = Metric('total_loss', -utility, batch.batch_size)
        metrics += total_loss

        return metrics
