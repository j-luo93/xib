"""
Analyzers package methods that are shared between evaluators and trainers.
"""

from typing import Dict, Tuple, Union

import torch

from dev_misc import FT, add_argument, g, get_tensor
from dev_misc.devlib import get_length_mask
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics
from xib.aligned_corpus.data_loader import AlignedIpaBatch
from xib.ipa import should_include
from xib.model.extract_model import ExtractModelReturn


class ExtractAnalyzer:

    add_argument('mean_mode', default='segment', choices=['segment', 'char'])
    add_argument('bij_mode', default='square', choices=['square', 'abs', 'sinkhorn'])
    add_argument('top_only', default=False, dtype=bool)

    def analyze(self, model_ret: ExtractModelReturn, batch: AlignedIpaBatch) -> Metrics:
        metrics = Metrics()
        if g.top_only:
            span_log_probs = model_ret.extracted.matches.ll.logsumexp(dim='vocab')
            loss_value = span_log_probs.max(dim='len_s')[0].max(dim='len_e')[0].sum(dim='batch')
        else:
            loss_value = model_ret.ctc_return.final_score.sum(dim='batch')
        not_zero = True
        loss_weight = batch.batch_size if g.mean_mode == 'segment' else batch.lengths.sum()
        loss_metric = Metric('marginal', loss_value, loss_weight)
        metrics += loss_metric

        almt = model_ret.costs.alignment

        def compute_bij(almt):
            almt = almt.sum(dim=0)
            almt = ((1.0 - almt).clamp_min(0.0) + (almt - 1.0).clamp_min(0.0)) ** 2
            reg = almt.sum() * float(not_zero)
            return reg

        if almt is not None:  # and not g.use_new_model:
            if True:  # not g.use_new_model:
                # distr = almt.lost2known if g.use_new_model else almt.known2lost
                distr = almt.known2lost
                if g.bij_mode == 'square':
                    bijective_reg = compute_bij(distr)
                    # bijective_reg = ((almt.known2lost.sum(dim=0) - 1.0).clamp_max(0.0) ** 2).sum() * float(not_zero)
                    # bijective_reg = ((almt.known2lost.sum(dim=0) - 1.0) ** 2).sum() * float(not_zero)
                    # bijective_reg = ((almt.known2lost.sum(dim=0)[1:] - 1.0) ** 2).sum() * float(not_zero)
                elif g.bij_mode == 'sinkhorn':
                    bijective_reg = distr
                else:
                    # bijective_reg = ((almt.known2lost.sum(dim=0)[1:] - 1.0).abs()).sum() * float(not_zero)
                    bijective_reg = ((distr.sum(dim=0) - 1.0).abs()).sum() * float(not_zero)
                metrics += Metric('bij_reg', bijective_reg, loss_weight)
            if g.use_entropy_reg:
                ent_k2l_reg = -(almt.known2lost * (1e-8 + almt.known2lost).log()).sum() * float(not_zero)
                ent_l2k_reg = -(almt.lost2known * (1e-8 + almt.lost2known).log()).sum() * float(not_zero)
                metrics += Metric('ent_k2l_reg', ent_k2l_reg, loss_weight)
                metrics += Metric('ent_l2k_reg', ent_l2k_reg, loss_weight)
            variance = almt.variance
            if variance is not None:
                metrics += Metric('variance', -variance, loss_weight)

        try:
            pr_reg = Metric('posterior_spans',
                            model_ret.ctc_return.expected_num_spans.sum('batch'),
                            batch.lengths.sum())
            metrics += pr_reg

            # FIXME(j_luo) The weight is wrongly computed.
            l_pr_reg = Metric('avg_log_probs',
                              model_ret.ctc_return.expected_avg_log_probs.sum('batch'),
                              loss_weight)
            metrics += l_pr_reg
        except:
            pass

        return metrics
