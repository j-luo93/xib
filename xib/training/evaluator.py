from __future__ import annotations

from dev_misc.utils import global_property
import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from dev_misc import add_argument, g
from dev_misc.arglib import g
from dev_misc.devlib import get_range, get_tensor
from dev_misc.devlib.named_tensor import NoName, get_named_range
from dev_misc.trainlib import Metric, Metrics
from dev_misc.trainlib.tracker.trackable import BaseTrackable
from dev_misc.trainlib.tracker.tracker import Tracker
from dev_misc.utils import deprecated, global_property, pbar
from xib.aligned_corpus.corpus import (AlignedSentence, Segment,
                                       UnsegmentedSentence)
from xib.aligned_corpus.data_loader import AlignedBatch, AlignedDataLoader
from xib.aligned_corpus.vocabulary import Vocabulary
from xib.data_loader import (ContinuousIpaBatch, ContinuousTextDataLoader,
                             DataLoaderRegistry)
from xib.ipa.process import Segmentation, Span
from xib.model.decipher_model import (DecipherModel, DecipherModelReturn,
                                      Segmentation, Span)
from xib.model.extract_model import ExtractModel, ExtractModelReturn
from xib.search.search_solver import SearchSolver
from xib.training.analyzer import DecipherAnalyzer, ExtractAnalyzer, LMAnalyzer
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
            total_num_samples = 0
            for batch in self.data_loader:
                if g.eval_max_num_samples and total_num_samples + batch.batch_size > g.eval_max_num_samples:
                    logging.imp(
                        f'Stopping at {total_num_samples} < {g.eval_max_num_samples} evaluated examples from.')
                    break

                scores = self.model.score(batch)
                try:
                    metrics = self.analyzer.analyze(scores.distr)
                except AttributeError:
                    metrics = self.analyzer.analyze(scores)
                all_metrics += metrics

                total_num_samples += batch.batch_size
        return all_metrics


def get_matching_stats(predictions: List[Segmentation], ground_truths: List[Segmentation], match_words: bool = False) -> Metrics:
    exact_span_matches = 0
    prefix_span_matches = 0
    if match_words:
        exact_word_matches = 0
        prefix_word_matches = 0
    for pred, gt in zip(predictions, ground_truths):
        for p in pred:
            for g in gt:
                if p.is_same_span(g):
                    exact_span_matches += 1
                    prefix_span_matches += 1
                    if match_words and p.is_same_word(g):
                        exact_word_matches += 1
                        prefix_word_matches += 1
                elif p.is_prefix_span_of(g) or g.is_prefix_span_of(p):
                    prefix_span_matches += 1
                    if match_words and (p.is_prefix_word_of(g) or g.is_prefix_word_of(p)):
                        prefix_word_matches += 1

    total_correct = sum(map(len, ground_truths))
    total_pred = sum(map(len, predictions))
    exact_span_matches = Metric(f'prf_exact_span_matches', exact_span_matches, 1.0, report_mean=False)
    prefix_span_matches = Metric(f'prf_prefix_span_matches', prefix_span_matches, 1.0, report_mean=False)
    total_correct = Metric(f'prf_total_correct', total_correct, 1.0, report_mean=False)
    total_pred = Metric(f'prf_total_pred', total_pred, 1.0, report_mean=False)
    metrics = Metrics(exact_span_matches, prefix_span_matches, total_correct, total_pred)

    if match_words:
        exact_word_matches = Metric(f'prf_exact_word_matches', exact_word_matches, 1.0, report_mean=False)
        prefix_word_matches = Metric(f'prf_prefix_word_matches', prefix_word_matches, 1.0, report_mean=False)
        metrics += Metrics(exact_word_matches, prefix_word_matches)
    return metrics


def get_prf_scores(metrics: Metrics) -> Metrics:
    prf_scores = Metrics()

    def _get_f1(p, r):
        return 2 * p * r / (p + r + 1e-8)

    exact_span_matches = getattr(metrics, f'prf_exact_span_matches').total
    prefix_span_matches = getattr(metrics, f'prf_prefix_span_matches').total
    total_pred = getattr(metrics, f'prf_total_pred').total
    total_correct = getattr(metrics, f'prf_total_correct').total
    exact_span_precision = exact_span_matches / (total_pred + 1e-8)
    exact_span_recall = exact_span_matches / (total_correct + 1e-8)
    exact_span_f1 = _get_f1(exact_span_precision, exact_span_recall)
    prefix_span_precision = prefix_span_matches / (total_pred + 1e-8)
    prefix_span_recall = prefix_span_matches / (total_correct + 1e-8)
    prefix_span_f1 = _get_f1(prefix_span_precision, prefix_span_recall)
    prf_scores += Metric(f'prf_exact_span_precision', exact_span_precision, report_mean=False)
    prf_scores += Metric(f'prf_exact_span_recall', exact_span_recall, 1.0, report_mean=False)
    prf_scores += Metric(f'prf_exact_span_f1', exact_span_f1, 1.0, report_mean=False)
    prf_scores += Metric(f'prf_prefix_span_precision', prefix_span_precision, report_mean=False)
    prf_scores += Metric(f'prf_prefix_span_recall', prefix_span_recall, 1.0, report_mean=False)
    prf_scores += Metric(f'prf_prefix_span_f1', prefix_span_f1, 1.0, report_mean=False)
    return prf_scores

# @deprecated


class DecipherEvaluator(BaseEvaluator):

    add_argument('eval_max_num_samples', default=0, dtype=int, msg='Max number of samples to evaluate on.')

    def __init__(self, model: DecipherModel, dl_reg: DataLoaderRegistry, tasks: Sequence[DecipherTask]):
        self.model = model
        self.dl_reg = dl_reg
        self.tasks = tasks
        self.analyzer = DecipherAnalyzer()

    def evaluate(self, stage: str) -> Metrics:
        metrics = Metrics()
        with torch.no_grad():
            self.model.eval()
            for task in self.tasks:
                dl = self.dl_reg[task]
                task_metrics = self._evaluate_one_data_loader(dl, stage)
                metrics += task_metrics.with_prefix_(task)
        return metrics

    def _evaluate_one_data_loader(self, dl: ContinuousTextDataLoader, stage: stage) -> Metrics:
        task = dl.task
        accum_metrics = Metrics()

        # Get all metrics from batches.
        dfs = list()
        total_num_samples = 0
        for batch in dl:
            if g.eval_max_num_samples and total_num_samples + batch.batch_size > g.eval_max_num_samples:
                logging.imp(
                    f'Stopping at {total_num_samples} < {g.eval_max_num_samples} evaluated examples from {task}.')
                break

            model_ret = self.model(batch)

            batch_metrics, batch_df = self.predict(model_ret, batch)
            accum_metrics += batch_metrics
            # accum_metrics += self.analyzer.analyze(model_ret, batch)
            total_num_samples += batch.batch_size
            dfs.append(batch_df)

        df = pd.concat(dfs, axis=0)
        # Write the predictions to file.
        out_path = g.log_dir / 'predictions' / f'{task}.{stage}.tsv'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_path, index=None, sep='\t')

        # Compute P/R/F scores.
        accum_metrics += get_prf_scores(accum_metrics)
        return accum_metrics

    def _get_predictions(self, model_ret: DecipherModelReturn, batch: ContinuousIpaBatch) -> List[Segmentation]:
        label_log_probs = model_ret.probs.label_log_probs.align_to('batch', 'length', 'label')
        _, tag_seqs = label_log_probs.max(dim='label')
        tag_seqs = tag_seqs.align_to('batch', 'sample', 'length').int()
        lengths = batch.lengths.align_to('batch', 'sample').int()
        segment_list = None
        if self.model.vocab is not None:
            segment_list = [segment.segment_list for segment in batch.segments]
        packed_words = self.model.pack(
            tag_seqs, lengths, batch.feat_matrix, batch.segments, segment_list=segment_list)
        segments_by_batch = packed_words.sampled_segments_by_batch
        # Only take the first (and only) sample.
        predictions = [segments[0] for segments in segments_by_batch]
        return predictions

    def predict(self, model_ret: DecipherModelReturn, batch: ContinuousIpaBatch) -> Tuple[Metrics, pd.DataFrame]:
        metrics = Metrics()
        predictions = self._get_predictions(model_ret, batch)
        ground_truths = [segment.to_segmentation() for segment in batch.segments]
        matching_stats = get_matching_stats(predictions, ground_truths)
        metrics += matching_stats

        df = _get_df(batch.segments, ground_truths, predictions)

        return metrics, df


def _get_df(*seqs: Sequence, columns=('segment', 'ground_truth', 'prediction')):
    data = map(lambda x: map(str, x), zip(*seqs))
    df = pd.DataFrame(data, columns=columns)
    return df


class SearchSolverEvaluator(BaseEvaluator):

    def __init__(self, solver: SearchSolver):
        self.solver = solver

    def evaluate(self, dl: ContinuousTextDataLoader) -> Metrics:
        segments = list()
        ground_truths = list()
        predictions = list()
        for batch in pbar(dl, desc='eval_batch'):
            for segment in batch.segments:
                segments.append(segment)
                ground_truth = segment.to_segmentation()
                ground_truths.append(ground_truth)

                best_value, best_state = self.solver.find_best(segment)
                prediction = Segmentation(best_state.spans)
                predictions.append(prediction)

        df = _get_df(segments, ground_truths, predictions)
        out_path = g.log_dir / 'predictions' / 'search_solver.tsv'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_path, index=None, sep='\t')
        matching_stats = get_matching_stats(predictions, ground_truths)
        prf_scores = get_prf_scores(matching_stats)
        return matching_stats + prf_scores


class ExtractEvaluator(BaseEvaluator):

    def __init__(self, model: ExtractModel, dl: ContinuousTextDataLoader):
        self.model = model
        self.dl = dl
        self.analyzer = ExtractAnalyzer()

    def evaluate(self, stage: str) -> Metrics:
        segments = list()
        predictions = list()
        ground_truths = list()
        matched_segments = list()
        total_num_samples = 0
        analyzed_metrics = Metrics()
        for batch in pbar(self.dl, desc='eval_batch'):

            if g.eval_max_num_samples and total_num_samples + batch.batch_size > g.eval_max_num_samples:
                logging.imp(
                    f'Stopping at {total_num_samples} < {g.eval_max_num_samples} evaluated examples.')
                break

            ret = self.model(batch)
            analyzed_metrics += self.analyzer.analyze(ret, batch)

            segments.extend(list(batch.segments))
            segmentations, _matched_segments = self._get_segmentations(ret, batch)
            predictions.extend(segmentations)
            matched_segments.extend(_matched_segments)
            ground_truths.extend([segment.to_segmentation() for segment in batch.segments])
            total_num_samples += batch.batch_size

        df = _get_df(segments, ground_truths, predictions, matched_segments,
                     columns=('segment', 'ground_truth', 'prediction', 'matched_segment'))
        out_path = g.log_dir / 'predictions' / f'extract.{stage}.tsv'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(out_path, index=None, sep='\t')
        matching_stats = get_matching_stats(predictions, ground_truths)
        prf_scores = get_prf_scores(matching_stats)
        return analyzed_metrics + matching_stats + prf_scores

    def _get_segmentations(self, model_ret: ExtractModelReturn, batch: ContinuousIpaBatch) -> Tuple[List[Segmentation], np.ndarray]:
        # Get the best matched nll.
        start = model_ret.start
        end = model_ret.end
        bmv = model_ret.top_matched_vocab[:, 0]
        bmnll, _ = -model_ret.top_matched_ll[:, 0]
        matched = bmnll < self.model.threshold

        start = start.cpu().numpy()
        end = end.cpu().numpy()
        bmv = bmv.cpu().numpy()
        bmw = self.model.vocab[bmv]  # Best matched word

        segmentations = list()
        matched_segments = list()
        for segment, s, e, m, w in zip(batch.segments, start, end, matched, bmw):
            spans = list()
            if len(segment) >= g.min_word_length and m:
                span = [segment[i] for i in range(s, e + 1)]
                span = Span('-'.join(span), s, e)
                spans.append(span)
                matched_segments.append(w)
            else:
                matched_segments.append('')
            segmentations.append(Segmentation(spans))
        return segmentations, matched_segments


def _get_prf_metrics(num_pred: int, num_gold: int, num_match: int, name: str) -> Metrics:
    ret = Metrics()
    precision = num_match / (num_pred + 1e-8)
    recall = num_match / (num_gold + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    ret += Metric(f'prf_{name}_total_correct', num_gold, report_mean=False)
    ret += Metric(f'prf_{name}_matches', num_match, report_mean=False)
    ret += Metric(f'prf_{name}_total_pred', num_pred, report_mean=False)
    ret += Metric(f'prf_{name}_precision', precision, report_mean=False)
    ret += Metric(f'prf_{name}_recall', recall, report_mean=False)
    ret += Metric(f'prf_{name}_f1', f1, report_mean=False)
    return ret


@dataclass
class _Match:
    # total_log_prob: float
    word_log_prob: float
    raw_word_log_prob: float
    avg_char_log_prob: float
    raw_avg_char_log_prob: float
    hypothesis: UnsegmentedSentence


@dataclass
class _AnnotationTuple:
    sentence: AlignedSentence
    gold: UnsegmentedSentence
    pred: UnsegmentedSentence
    # unmatched_log_prob: Optional[float] = None
    top_matched: Optional[List[_Match]] = field(default_factory=list)


class AlignedExtractEvaluator(BaseEvaluator):

    add_argument('evaluate_baselines', nargs='+', dtype=float, default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    def __init__(self, model: ExtractModel, dl: AlignedDataLoader, vocab: Vocabulary):
        self.model = model
        self.dl = dl
        self.vocab = vocab.vocab
        self.analyzer = ExtractAnalyzer()
        self._last_eval_stage: str = None
        self._last_annotations_results: List[_AnnotationTuple] = None

    def get_best_spans(self, stage: str, total_num_spans: int) -> Tuple[Sequence[AlignedSentence], Sequence[Tuple[int, int]]]:
        self.model.eval()
        with torch.no_grad():
            _, annotations = self._evaluate_core(stage)

        candidates = list()
        for i, anno in enumerate(annotations):
            for match in anno.top_matched:
                candidates.append((match.raw_avg_char_log_prob, i, match.hypothesis.segments[0].single_segments[0]))
        candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:total_num_spans]
        # avg_char_log_probs = [anno.top_matched[0].avg_char_log_prob for anno in annotations]
        # aclp = np.asarray(avg_char_log_probs)
        # best_idx = np.argsort(-aclp)[:total_num_spans]
        sentences = list()
        spans = list()
        # for idx in best_idx:
        for avg, idx, seg in candidates:
            anno = annotations[idx]
            # try:
            #     seg = anno.pred.segments[0]
            # except IndexError:
            #     continue
            sentences.append(anno.sentence)
            spans.append((seg.start, seg.end))
        return sentences, spans

    @global_property
    def global_baseline(self):
        pass

    @global_baseline.setter
    def global_baseline(self, value):
        pass

    def _evaluate_core(self, stage: str) -> Tuple[Metrics, List[_AnnotationTuple]]:
        if self._last_eval_stage is not None and self._last_eval_stage == stage:
            return self._last_annotations_results

        total_num_samples = 0
        analyzed_metrics = Metrics()
        annotations = list()
        orig_baseline = self.global_baseline
        baselines = [None] + list(g.evaluate_baselines) if g.anneal_baseline else [None]
        for baseline in baselines:
            if baseline is not None:
                self.global_baseline = baseline
                self.model.train()
            for batch in pbar(self.dl, desc='eval_batch'):
                if g.eval_max_num_samples and total_num_samples + batch.batch_size > g.eval_max_num_samples:
                    logging.imp(
                        f'Stopping at {total_num_samples} < {g.eval_max_num_samples} evaluated examples.')
                    break
                ret = self.model(batch)
                batch_metrics = self.analyzer.analyze(ret, batch)
                if baseline is not None:
                    batch_metrics = batch_metrics.with_prefix_(f'eval_b{baseline}')
                analyzed_metrics += batch_metrics
                total_num_samples += batch.batch_size
                if baseline is None:
                    annotations.extend(self._get_annotations_for_batch(ret, batch))
        self.model.eval()
        self.global_baseline = orig_baseline
        self._last_eval_stage = stage
        self._last_annotations_results = (analyzed_metrics, annotations)
        return analyzed_metrics, annotations

    def evaluate(self, stage: str) -> Metrics:
        if g.cut_off is not None:
            # HACK(j_luo)
            try:
                self._cnt += 1
            except AttributeError:
                self._cnt = 0
            self.cut_off = -g.cut_off[self._cnt]

            logging.imp(f'cut off is set to {self.cut_off}')

        analyzed_metrics, annotations = self._evaluate_core(stage)

        # Write to file.
        data = list()
        for anno in annotations:
            segmented_content = anno.gold.segmented_content
            g_seg_str = '&'.join(map(str, anno.gold.segments))
            p_seg_str = '&'.join(map(str, anno.pred.segments))
            top_matched_strings = list()
            for match in anno.top_matched:
                # log_prob_str = f'{match.total_log_prob:.3f}'
                word_log_prob_str = f'{match.word_log_prob:.3f}'
                raw_word_log_prob_str = f'{match.raw_word_log_prob:.3f}'
                avg_char_log_prob_str = f'{match.avg_char_log_prob:.3f}'
                raw_avg_char_log_prob_str = f'{match.raw_avg_char_log_prob:.3f}'
                segments_str = '&'.join(map(str, match.hypothesis.segments))
                # top_matched_strings.append(
                #     f'({log_prob_str}, {word_log_prob_str}, {avg_char_log_prob_str}, {segments_str})')
                top_matched_strings.append(
                    f'({word_log_prob_str}, {raw_word_log_prob_str}, {avg_char_log_prob_str}, {raw_avg_char_log_prob_str}, {segments_str})')
            top_matched_seg_str = ', '.join(top_matched_strings)
            # try:
            #     um = f'{anno.unmatched_log_prob:.3f}'
            # except TypeError:
            #     um = None
            # data.append((segmented_content, g_seg_str, p_seg_str, um, top_matched_seg_str))
            data.append((segmented_content, g_seg_str, p_seg_str, top_matched_seg_str))
        # out_df = pd.DataFrame(data, columns=['segmented_content', 'gold',
        #                                      'predicted', 'unmatched_log_prob', 'top_matched'])
        out_df = pd.DataFrame(data, columns=['segmented_content', 'gold',
                                             'predicted', 'top_matched'])
        segment_df = pd.DataFrame({
            'gold': [anno.gold.segments for anno in annotations],
            'pred': [anno.pred.segments for anno in annotations]})
        segment_df = segment_df.reset_index().rename(columns={'index': 'segment_idx'})
        flat_segment_df = segment_df.explode('gold').reset_index(drop=True).explode('pred')

        def safe_bifunc(func):

            def wrapped(item):
                x, y = item
                if pd.isnull(x) or pd.isnull(y):
                    return None
                return func(x, y)

            return wrapped

        # Get P/R/F scores.
        # flat_segment_df['exact_span_match'] = flat_segment_df[['pred', 'gold']].apply(safe_bifunc(Segment.has_same_span),
        #                                                                               axis=1)
        # flat_segment_df['prefix_span_match'] = flat_segment_df[['pred', 'gold']].apply(safe_bifunc(Segment.has_prefix_span),
        #                                                                                axis=1)
        flat_segment_df['exact_content_match'] = flat_segment_df[['pred', 'gold']].apply(safe_bifunc(Segment.has_correct_prediction),
                                                                                         axis=1)

        flat_segment_df = flat_segment_df.fillna(value=False)
        segment_score_df = flat_segment_df.pivot_table(index='segment_idx',
                                                       #    values=['exact_span_match',
                                                       #            'prefix_span_match', 'exact_content_match'],
                                                       values=['exact_content_match'],
                                                       aggfunc=np.sum)
        segment_df = pd.merge(segment_df, segment_score_df, left_on='segment_idx', right_index=True, how='left')
        out_df['num_gold'] = segment_df['gold'].apply(len)
        out_df['num_pred'] = segment_df['pred'].apply(len)
        has_cognate = segment_df['gold'].apply(bool)
        has_cognate_int = has_cognate.apply(int)
        out_df['num_positive_gold'] = out_df['num_gold'] * has_cognate_int
        out_df['num_positive_pred'] = out_df['num_pred'] * has_cognate_int
        out_df['num_exact_content_match'] = segment_df['exact_content_match'].apply(int)
        out_df['num_positive_exact_content_match'] = out_df['num_exact_content_match'] * has_cognate_int

        out_path = g.log_dir / 'predictions' / f'aligned.{stage}.tsv'
        out_path.parent.mkdir(exist_ok=True, parents=True)
        out_df.to_csv(out_path, index=None, sep='\t')

        sums = out_df[['num_pred', 'num_gold', 'num_exact_content_match',
                       'num_positive_gold', 'num_positive_pred', 'num_positive_exact_content_match']].sum()
        exact_content_prf_scores = _get_prf_metrics(sums.num_pred,
                                                    sums.num_gold,
                                                    sums.num_exact_content_match,
                                                    'exact_content')
        positive_exact_content_prf_scores = _get_prf_metrics(sums.num_positive_pred,
                                                             sums.num_positive_gold,
                                                             sums.num_positive_exact_content_match,
                                                             'positive_exact_content')
        # return analyzed_metrics + exact_span_prf_scores + prefix_span_prf_scores + exact_content_prf_scores + exact_positive_content_prf_scores
        return analyzed_metrics + exact_content_prf_scores + positive_exact_content_prf_scores

    @global_property
    def cut_off(self):
        pass

    @cut_off.setter
    def cut_off(self, value):
        pass

    def _get_annotations_for_batch(self, model_ret: ExtractModelReturn, batch: AlignedBatch) -> List[_AnnotationTuple]:
        bkp = model_ret.ctc_return.bookkeeper
        bpt = bkp.best_prev_tags
        max_length = batch.lengths.max().item()

        final_nodes = model_ret.ctc_return.final_nodes
        prev_tag = last_tag = final_nodes.max(dim=1)[1]
        prev_pos = last_pos = batch.lengths
        offset = torch.LongTensor([1] + list(range(g.min_word_length, g.max_word_length + 1)))
        offset = get_tensor(offset)
        offset.rename_('tag')
        bs = final_nodes.size('batch')
        hyps = list()
        for l in range(max_length, 0, -1):
            with NoName(bpt[l], prev_tag):
                new_prev_tag = bpt[l][range(bs), prev_tag]
            new_prev_pos = l - offset.gather('tag', prev_tag)

            is_last = l == batch.lengths
            prev_pos = torch.where(is_last, last_pos, prev_pos)
            prev_tag = torch.where(is_last, last_tag, prev_tag)

            to_update = l == prev_pos
            prev_pos = torch.where(to_update, new_prev_pos, prev_pos)
            prev_tag = torch.where(to_update, new_prev_tag, prev_tag)
            hyps.append((prev_pos, prev_tag))
        pos, tag = zip(*reversed(hyps))
        pos = torch.stack(pos, new_name='length').cpu().numpy()
        tag = torch.stack(tag, new_name='length').cpu().numpy()
        last_pos = last_pos.cpu().numpy()
        last_tag = last_tag.cpu().numpy()

        lens = batch.lengths.cpu().numpy()
        tag_seqs = list()
        for p, t, lp, lt, l in zip(pos, tag, last_pos, last_tag, lens):
            p = np.concatenate([p[:l + 1], lp.reshape(1)], axis=0)
            t = np.concatenate([t[:l + 1], lt.reshape(1)], axis=0)
            last_p = 0
            i = 0
            tag_seq = list()
            while i < len(p):
                while i < len(p) and p[i] == last_p:
                    i += 1
                if i == len(p):
                    break
                tag_seq.append((p[i], t[i]))
                last_p = p[i]
            tag_seqs.append(tag_seq)

        offset = offset.cpu().numpy()
        best_vocab = bkp.best_vocab.cpu().numpy()
        ret = list()
        is_lost_ipa = (g.input_format == 'ipa')
        log_probs = model_ret.extracted.matches.ll.cpu()
        raw_log_probs = model_ret.extracted.matches.raw_ll.cpu()
        for i, (sentence, tag_seq, bv) in enumerate(zip(batch.sentences, tag_seqs, best_vocab)):
            gold = sentence.to_unsegmented(is_lost_ipa=is_lost_ipa, is_known_ipa=True, annotated=True)
            pred = sentence.to_unsegmented(is_lost_ipa=is_lost_ipa, is_known_ipa=True, annotated=False)

            top_matches = list()
            for pos, tag in tag_seq:
                if tag != 0:
                    length = offset[tag]
                    end = pos - 1
                    start = end - length + 1
                    end_idx = end - start - g.min_word_length + 1
                    start_idx = start
                    v_idx = bv[start_idx, end_idx]
                    y = self.vocab[v_idx]
                    lp = log_probs[i, start_idx, end_idx, v_idx].item()
                    raw_lp = raw_log_probs[i, start_idx, end_idx, v_idx].item()
                    avg_lp = lp / length
                    raw_avg_lp = raw_lp / length

                    # if raw_avg_lp < 0.6:
                    #     warnings.warn('Only taking most confident ones.')
                    #     continue

                    pred.annotate([start], [end], y)

                    uss = sentence.to_unsegmented(is_lost_ipa=is_lost_ipa, is_known_ipa=True, annotated=False)
                    uss.annotate([start], [end], y)
                    match = _Match(lp, raw_lp, avg_lp, raw_avg_lp, uss)
                    top_matches.append(match)

            annotation_tuple = _AnnotationTuple(sentence, gold, pred, top_matches)
            ret.append(annotation_tuple)
        return ret
