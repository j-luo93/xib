"""
Analyzers package methods that are shared between evaluators and trainers.
"""

from typing import Dict, Tuple, Union

import torch

from dev_misc import FT, g, get_tensor
from dev_misc.devlib import get_length_mask
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics
from xib.data_loader import ContinuousIpaBatch
from xib.ipa import should_include
from xib.model.decipher_model import DecipherModel, DecipherModelReturn
from xib.model.extract_model import ExtractModelReturn
from xib.model.lm_model import AdaptLMReturn, Cat


class LMAnalyzer:

    def analyze(self, scores: Dict[Cat, FT], return_scores: bool = False) -> Union[Metrics, Tuple[Metrics, Dict[Cat, FT]]]:
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
        if return_scores:
            return metrics, scores
        else:
            return metrics


class AdaptLMAnalyzer(LMAnalyzer):

    def analyze(self, ret: AdaptLMReturn) -> Metrics:
        metrics, scores = super().analyze(ret.distr, return_scores=True)
        # if g.use_moe:
        #     # prior = get_tensor([g.prior_value, 1.0 - g.prior_value]).squeeze(dim=0)
        #     # lp = ret.gate_log_probs
        #     # kld = lp.exp() * (lp - prior.log())

        #     # kld.
        #     lp = ret.gate_log_probs
        #     _p = lp.exp().sum(0) / lp.exp().sum()
        #     prior = get_tensor([g.prior_value, 1.0 - g.prior_value]).squeeze(dim=0)
        #     kld = _p * (_p.log() - prior.log())

        #     bs = lp.size('batch')
        #     kld = Metric('kld', kld.sum() * bs, bs)
        #     metrics += kld

        #     # sparsity.
        #     _p = lp.exp()
        #     with NoName(_p):
        #         # sparsity = torch.nn.functional.softmin(torch.stack([_p, 1.0 - _p], dim=-1), dim=-1)
        #         sparsity = torch.min(_p, 1.0 - _p)
        #     sparsity = Metric('sparsity', sparsity.sum(), bs)
        #     metrics += sparsity

        #     metrics.rename('loss', 'ce_loss')
        #     metrics += Metric('loss', metrics.ce_loss.total, bs)
        #     # metrics += Metric('loss', metrics.ce_loss.total + kld.total, bs)
        #     # metrics += Metric('loss', metrics.ce_loss.total + kld.total + sparsity.total, bs)

        if g.use_moe:
            metrics = Metrics()
            metrics_noise, scores_noise = super().analyze(ret.distr_noise, return_scores=True)
            total_loss = 0.0
            total_weight = 0.0
            cnt = 0
            prob_cnt = 0

            # gate_log_probs = ret.gate_logits.log_softmax(dim=-1)

            all_scores = [s for _, (s, _) in scores.items()]
            all_weights = [w for _, (_, w) in scores.items()]
            weight = all_weights[0]

            sum_scores = torch.stack(all_scores, new_name='stacked').sum(dim='stacked')
            batch_probs = ret.gate_logits.log_softmax(dim=-1).exp()[:, 0] * weight  # + (-999.9) * (1.0 - weight))
            # batch_probs = (ret.gate_logits[:, 0] + (-999.9) * (1.0 - weight)).log_softmax(dim='batch').exp()
            bs = batch_probs.size('batch')
            total = int(g.prior_value * weight.sum())
            diff_loss = ((batch_probs.sum() - total) ** 2).sum()
            diff_loss = Metric('diff_loss', diff_loss, bs)
            loss = (sum_scores * batch_probs).sum()
            loss = Metric('loss', loss + diff_loss.total, bs)

            metrics += diff_loss
            metrics += loss

            # for name in scores:
            #     s, w = scores[name]
            #     sn, _ = scores_noise[name]
            #     all_score = torch.stack([s, sn], new_name='expert')
            #     probs = gate_log_probs.exp()
            #     loss = ((all_score * probs) * w.align_as(all_score)).sum()
            #     cnt += ((all_score[:, 0] < all_score[:, 1]) * w).sum()
            #     prob_cnt += ((probs[:, 0] > probs[:, 1]) * w).sum()
            #     weight = w.sum()
            #     total_loss += loss
            #     total_weight += weight
            #     loss = Metric(f'loss_{name.snake}', loss, weight)
            #     metrics += loss

            # # kld.
            # lp = gate_log_probs
            # _p = lp.exp().sum(0) / lp.exp().sum()
            # prior = get_tensor([g.prior_value, 1.0 - g.prior_value]).squeeze(dim=0)
            # kld = _p * (_p.log() - prior.log())

            # bs = lp.size('batch')
            # kld = Metric('kld', kld.sum() * bs, bs)
            # metrics += kld

            # metrics += Metric('loss', total_loss, total_weight)
            # metrics += Metric('loss', total_loss + kld.total, total_weight)

            # print('cnt', cnt / total_weight)
            # print('prob', prob_cnt / total_weight)
            return metrics
        else:
            return metrics


def _compute_utility(logits: FT, sample_scores: FT) -> FT:
    sample_log_probs = logits.log_softmax(dim='sample')
    utility = (sample_log_probs.exp() * sample_scores).sum()
    return utility


class DecipherAnalyzer:

    def analyze(self, model_ret: DecipherModelReturn, batch: ContinuousIpaBatch) -> Metrics:
        if g.supervised:
            return self._analyze_supervised(model_ret, batch)
        else:
            return self._analyze_unsupervised(model_ret, batch)

    def _analyze_supervised(self, model_ret: DecipherModelReturn, batch: ContinuousIpaBatch) -> Metrics:
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

    def _analyze_unsupervised(self, model_ret: DecipherModelReturn, batch: ContinuousIpaBatch) -> Metrics:
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


class ExtractAnalyzer:

    def analyze(self, model_ret: ExtractModelReturn, batch: ContinuousIpaBatch) -> Metrics:
        metrics = Metrics()
        metrics += Metric('ll', model_ret.best_matched_ll.sum(), batch.batch_size)

        almt = model_ret.alignment
        if almt is not None:
            reg = ((almt.sum(dim=0) - 1.0) ** 2).sum()
            metrics += Metric('reg', reg, batch.batch_size)
        return metrics
