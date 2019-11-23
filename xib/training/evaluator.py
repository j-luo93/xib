from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence

import pandas as pd
import torch

from dev_misc.arglib import init_g_attr
from dev_misc.trainlib import Metric, Metrics
from xib.data_loader import ContinuousTextIpaBatch
from xib.ipa.process import Segmentation, Span
from xib.model.decipher_model import DecipherModel, Segmentation, Span
from xib.training.runner import BaseDecipherRunner, BaseLMRunner


class BaseEvaluator(ABC):

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    @abstractmethod
    def evaluate(self):
        pass


class LMEvaluator(BaseEvaluator, BaseLMRunner):
    """An evaluator class for LMs. Note that this is done over the entire training corpus, instead of a separate split."""

    def __init__(self, model, data_loader):
        super().__init__(model, data_loader)

    def evaluate(self) -> Metrics:
        with torch.no_grad():
            self.model.eval()
            all_metrics = Metrics()
            for batch in self.data_loader:
                scores = self.model.score(batch)
                metrics = self.analyze_scores(scores)
                all_metrics += metrics
        return all_metrics


@dataclass
class PrfScores:
    exact_matches: int
    total_correct: int
    total_pred: int

    @property
    def precision(self):
        return self.exact_matches / (self.total_pred + 1e-8)

    @property
    def recall(self):
        return self.exact_matches / (self.total_correct + 1e-8)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + 1e-8)

    def __add__(self, other: PrfScores) -> PrfScores:
        return PrfScores(self.exact_matches + other.exact_matches, self.total_correct + other.total_correct, self.total_pred + other.total_pred)


def get_prf_scores(predictions: List[Segmentation], ground_truths: List[Segmentation], mode: str) -> Metrics:
    exact_matches = 0
    for pred, gt in zip(predictions, ground_truths):
        for p in pred:
            for g in gt:
                if p == g:
                    exact_matches += 1
    total_correct = sum(map(len, ground_truths))
    total_pred = sum(map(len, predictions))
    exact_matches = Metric(f'prf_{mode}_exact_matches', exact_matches, 1.0, report_mean=False)
    total_correct = Metric(f'prf_{mode}_total_correct', total_correct, 1.0, report_mean=False)
    total_pred = Metric(f'prf_{mode}_total_pred', total_pred, 1.0, report_mean=False)
    return Metrics(exact_matches, total_correct, total_pred)


class DecipherEvaluator(LMEvaluator, BaseDecipherRunner):

    def evaluate(self) -> Metrics:
        with torch.no_grad():
            self.model.eval()
            accum_metrics = Metrics()
            for batch in self.data_loader:
                accum_metrics += self.predict(batch, ['local', 'global'])
                accum_metrics += self.get_metrics(batch)
            for mode in ['local', 'global']:
                exact_matches = getattr(accum_metrics, f'prf_{mode}_exact_matches').total
                total_pred = getattr(accum_metrics, f'prf_{mode}_total_pred').total
                total_correct = getattr(accum_metrics, f'prf_{mode}_total_correct').total
                precision = exact_matches / (total_pred + 1e-8)
                recall = exact_matches / (total_correct + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                # HACK(j_luo) Should use report_mean = True, but `save` cannot handle .total right now.
                accum_metrics += Metric(f'prf_{mode}_precision', precision, 1.0)
                accum_metrics += Metric(f'prf_{mode}_recall', recall, 1.0)
                accum_metrics += Metric(f'prf_{mode}_f1', f1, 1.0)
        return accum_metrics

    def predict(self, batch: ContinuousTextIpaBatch, modes: List[str]) -> Metrics:
        self.model: DecipherModel
        ret = self.model(batch)
        metrics = Metrics()
        for mode in modes:
            if mode == 'local':
                label_log_probs = ret['label_log_probs'].align_to('batch', 'length', 'label')
                _, tag_seqs = label_log_probs.max(dim='label')
                tag_seqs = tag_seqs.align_to('batch', 'sample', 'length').int()
                lengths = batch.lengths.align_to('batch', 'sample').int()
                packed_words, is_unique = self.model.pack(tag_seqs, lengths, batch.feat_matrix, batch.segments)
                segments_by_batch = packed_words.sampled_segments_by_batch
                # Only take the first (and only) sample.
                predictions = [segments[0] for segments in segments_by_batch]
            elif mode == 'global':
                seq_log_probs = ret['seq_log_probs']
                _, best_sample_inds = seq_log_probs.align_to('batch', 'sample').max(dim='sample')
                packed_words = ret['packed_words']
                segments_by_batch = packed_words.sampled_segments_by_batch
                predictions = [segments[best_sample_ind]
                               for segments, best_sample_ind in zip(segments_by_batch, best_sample_inds)]
            else:
                raise ValueError(f'Unrecognized value for mode "{mode}".')
            ground_truths = [segment.to_segmentation() for segment in batch.segments]
            prf_scores = get_prf_scores(predictions, ground_truths, mode)
            metrics += prf_scores

        return metrics
