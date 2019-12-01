from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from dev_misc import add_argument, g
from dev_misc.arglib import g
from dev_misc.devlib import get_range
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.tracker.trackable import BaseTrackable
from dev_misc.trainlib.tracker.tracker import Tracker
from dev_misc.utils import deprecated
from xib.data_loader import (ContinuousTextDataLoader, ContinuousTextIpaBatch,
                             DataLoaderRegistry)
from xib.ipa.process import Segmentation, Span
from xib.model.decipher_model import (DecipherModel, DecipherModelReturn,
                                      Segmentation, Span)
from xib.training.analyzer import DecipherAnalyzer, LMAnalyzer
from xib.training.task import DecipherTask


class BaseEvaluator(ABC):

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    @abstractmethod
    def evaluate(self): ...


class LMEvaluator(BaseEvaluator):
    """An evaluator class for LMs. Note that this is done over the entire training corpus, instead of a separate split."""

    def __init__(self, model, data_loader):
        super().__init__(model, data_loader)
        self.analyzer = LMAnalyzer()

    def evaluate(self, *args) -> Metrics:  # HACK(j_luo) *args is used just comply with BaseTrainer function signature.
        with torch.no_grad():
            self.model.eval()
            all_metrics = Metrics()
            for batch in self.data_loader:
                scores = self.model.score(batch)
                metrics = self.analyzer.analyze(scores)
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


def get_prf_scores(predictions: List[Segmentation], ground_truths: List[Segmentation]) -> Metrics:
    exact_matches = 0
    for pred, gt in zip(predictions, ground_truths):
        for p in pred:
            for g in gt:
                if p == g:
                    exact_matches += 1
    total_correct = sum(map(len, ground_truths))
    total_pred = sum(map(len, predictions))
    exact_matches = Metric(f'prf_exact_matches', exact_matches, 1.0, report_mean=False)
    total_correct = Metric(f'prf_total_correct', total_correct, 1.0, report_mean=False)
    total_pred = Metric(f'prf_total_pred', total_pred, 1.0, report_mean=False)
    return Metrics(exact_matches, total_correct, total_pred)


# @deprecated
class DecipherEvaluator(BaseEvaluator):

    add_argument('eval_max_num_samples', default=0, dtype=int, msg='Max number of samples to evaluate on.')

    def __init__(self, model: DecipherModel, dl_reg: DataLoaderRegistry, tasks: Sequence[DecipherTask]):
        self.model = model
        self.dl_reg = dl_reg
        self.tasks = tasks
        self.analyzer = DecipherAnalyzer()

    def evaluate(self, tracker: Tracker) -> Metrics:
        metrics = Metrics()
        with torch.no_grad():
            self.model.eval()
            for task in self.tasks:
                dl = self.dl_reg[task]
                task_metrics = self._evaluate_one_data_loader(dl, tracker)
                metrics += task_metrics.with_prefix_(task)
        return metrics

    def _evaluate_one_data_loader(self, dl: ContinuousTextDataLoader, tracker: Tracker) -> Metrics:
        task = dl.task
        accum_metrics = Metrics()

        # Get all metrics from batches.
        dfs = list()
        total_num_samples = 0
        for batch in dl:
            if g.eval_max_num_samples and total_num_samples + batch.batch_size > g.eval_max_num_samples:
                logging.imp(f'Stopping at {total_num_samples} < {g.eval_max_num_samples} evaluated examples.')
                break

            model_ret = self.model(batch)

            batch_metrics, batch_df = self.predict(model_ret, batch)
            accum_metrics += batch_metrics
            accum_metrics += self.analyzer.analyze(model_ret, batch)
            total_num_samples += batch.batch_size
            dfs.append(batch_df)

        df = pd.concat(dfs, axis=0)
        # Write the predictions to file.
        out_path = g.log_dir / 'predictions' / f'{task}.{tracker.total_step}.tsv'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_path, index=None, sep='\t')

        # Compute P/R/F scores.
        exact_matches = getattr(accum_metrics, f'prf_exact_matches').total
        total_pred = getattr(accum_metrics, f'prf_total_pred').total
        total_correct = getattr(accum_metrics, f'prf_total_correct').total
        precision = exact_matches / (total_pred + 1e-8)
        recall = exact_matches / (total_correct + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accum_metrics += Metric(f'prf_precision', precision, report_mean=False)
        accum_metrics += Metric(f'prf_recall', recall, 1.0, report_mean=False)
        accum_metrics += Metric(f'prf_f1', f1, 1.0, report_mean=False)
        return accum_metrics

    def _get_predictions(self, model_ret: DecipherModelReturn, batch: ContinuousTextIpaBatch) -> List[Segmentation]:
        label_log_probs = model_ret.probs.label_log_probs.align_to('batch', 'length', 'label')
        _, tag_seqs = label_log_probs.max(dim='label')
        tag_seqs = tag_seqs.align_to('batch', 'sample', 'length').int()
        lengths = batch.lengths.align_to('batch', 'sample').int()
        segment_list = None
        if self.model.vocab is not None:
            segment_list = [segment.segment_list for segment in batch.segments]
        packed_words, _ = self.model.pack(
            tag_seqs, lengths, batch.feat_matrix, batch.segments, segment_list=segment_list)
        segments_by_batch = packed_words.sampled_segments_by_batch
        # Only take the first (and only) sample.
        predictions = [segments[0] for segments in segments_by_batch]
        return predictions

    def predict(self, model_ret: DecipherModelReturn, batch: ContinuousTextIpaBatch) -> Tuple[Metrics, pd.DataFrame]:
        metrics = Metrics()
        predictions = self._get_predictions(model_ret, batch)
        ground_truths = [segment.to_segmentation() for segment in batch.segments]
        prf_scores = get_prf_scores(predictions, ground_truths)
        metrics += prf_scores

        data = map(lambda x: map(str, x), zip(batch.segments, ground_truths, predictions))
        df = pd.DataFrame(data, columns=['segment', 'ground_truth', 'prediction'])

        return metrics, df


# class DecipherEvaluator(OldDecipherEvaluator):

#     def _predict_with_mode(self, ret, batch, mode):
#         if g.search:
#             search_result = self.searcher.search(batch)
#             rs = search_result['readable_score']
#             urs = search_result['unreadable_score']
#             lms = search_result['lm_score']
#             ivs = search_result['in_vocab_score']
#             ds = search_result['diff_score']
#             ts = search_result['tag_score']
#             fv = torch.stack([rs, urs, lms, ivs, ds, ts], new_name='feature')
#             score = self.model.wv(fv).squeeze('score')
#             values, indices = score.max('sample')

#             max_len = batch.gold_tag_seqs.size('length')
#             length = batch.lengths.align_to('batch', 'length') - get_range(max_len, 2, 1) - 1
#             radical = torch.full_like(length, 3).pow(length)
#             _indices = indices.clone()
#             vs = list()
#             for l in range(max_len):
#                 _rad = radical[:, l]
#                 v = _indices // _rad
#                 _indices = _indices % _rad
#                 vs.append(v)
#             samples = torch.stack(vs, new_name='length').cpu().numpy().tolist()
#             predictions = list()
#             for sample, segment in zip(samples, batch.segments):
#                 seg = segment.get_segmentation_from_tags(sample)
#                 predictions.append(seg)
#             return predictions
#             # gold_sample_idx = (radical * batch.gold_tag_seqs).sum(dim='length')
#         if mode == 'risk':
#             return super()._predict_with_mode(ret, batch, 'local')
#         else:
#             return super()._predict_with_mode(ret, batch, mode)

#     @property
#     def model_mode(self):
#         return 'risk'

#     @property
#     def available_modes(self):
#         return ['risk']
