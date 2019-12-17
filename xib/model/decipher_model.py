import math
from collections import defaultdict
from dataclasses import InitVar, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from dev_misc.arglib import add_argument, g, not_supported_argument_value
from dev_misc.devlib import (BaseBatch, batch_class, dataclass_numpy,
                             dataclass_size_repr, get_array, get_tensor,
                             get_zeros)
from dev_misc.devlib.named_tensor import (NoName, expand_as, get_named_range,
                                          self_attend)
from dev_misc.trainlib import freeze
from dev_misc.utils import cached_property, deprecated
from xib.check_in_vocab_impl import check_in_vocab
from xib.data_loader import ContinuousIpaBatch, IpaBatch
from xib.extract_words_impl import extract_words_v8 as extract_words
from xib.gumbel import gumbel_softmax
from xib.ipa import Category, should_include
from xib.ipa.process import B, I, O, Segmentation, SegmentWindow, Span
from xib.model.modules import PositionalEmbedding, Predictor, TransformerLayer
from xib.search.searcher import BaseSearcher, BeamSearcher, BruteForceSearcher

from . import BT, FT, LT
from .lm_model import LM
from .modules import AdaptLayer, FeatEmbedding


@dataclass
class PackedWords:
    word_feat_matrices: LT
    word_lengths: LT
    batch_indices: LT
    sample_indices: LT
    word_positions: LT
    is_unique: BT
    num_samples: int
    orig_segments: np.ndarray
    in_vocab: Optional[BT]

    __repr__ = dataclass_size_repr
    numpy = dataclass_numpy

    @property
    def batch_size(self):
        return len(self.orig_segments)

    def __len__(self):
        return len(self.word_feat_matrices)

    @cached_property
    def segments(self) -> np.ndarray:
        ret = list()
        for bi in self.batch_indices.cpu().numpy():
            ret.append(self.orig_segments[bi])
        return get_array(ret)

    @cached_property
    def sampled_segments(self) -> np.ndarray:
        pw = self.numpy()
        ret = list()
        for bi, si, wps, wl in zip(pw.batch_indices, pw.sample_indices, pw.word_positions, pw.word_lengths):
            chars = [self.orig_segments[bi][wp] for i, wp in enumerate(wps) if i < wl]
            ret.append('-'.join(chars))
        return get_array(ret)

    @cached_property
    def sampled_segments_by_batch(self) -> List[List[Segmentation]]:
        pw = self.numpy()
        all_words = [defaultdict(list) for _ in range(self.batch_size)]
        for bi, si, wps, wl in zip(pw.batch_indices, pw.sample_indices, pw.word_positions, pw.word_lengths):
            chars = list()
            start = float('inf')
            end = -float('inf')
            for i, wp in enumerate(wps):
                if i < wl:
                    start = min(wp, start)
                    end = max(wp, end)
            assert start != float('inf') and end != -float('inf')
            chars = [self.orig_segments[bi][pos] for pos in range(start, end + 1)]
            span = Span('-'.join(chars), start, end)
            all_words[bi][si].append(span)
        ret = list()
        for bi in range(self.batch_size):
            segmentations = [Segmentation(all_words[bi][si]) for si in range(self.num_samples)]
            ret.append(segmentations)
        return ret

    def get_segment_nlls(self, nlls: FT) -> Tuple[str, float]:
        segments = self.sampled_segments
        return tuple(zip(segments, nlls.cpu().numpy()))


class EmptyPackedWords(Exception):
    pass


@batch_class
class DecipherModelProbReturn(BaseBatch):
    label_log_probs: FT
    sample_log_probs: FT


@batch_class
class DecipherModelScoreReturn(BaseBatch):
    lm_score: FT
    word_score: FT
    in_vocab_score: FT
    readable_score: FT
    unreadable_score: FT
    phi_score: FT


@batch_class
class DecipherModelReturn(BaseBatch):
    state: FT  # The hidden state before label prediction.
    probs: DecipherModelProbReturn
    packed_words: PackedWords
    ptb_packed_words: PackedWords  # NOTE(j_luo) ptb stands for perturbed.
    scores: DecipherModelScoreReturn
    ptb_scores: DecipherModelScoreReturn
    duplicates: List[bool]


class DecipherModel(nn.Module):

    add_argument('adapt_mode', default='none', choices=['none'], dtype=str,
                 msg='how to adapt the features from one language to another')
    add_argument('num_self_attn_layers', default=2, dtype=int, msg='number of self attention layers')
    add_argument('num_samples', default=100, dtype=int, msg='number of samples per sequence')
    add_argument('num_heads', default=4, dtype=int, msg='Number for heads for self attention.')
    add_argument('lm_model_path', dtype='path', msg='path to a pretrained lm model')
    add_argument('dropout', default=0.0, dtype=float, msg='dropout rate')
    add_argument('sampling_temperature', default=1.0, dtype=float, msg='Sampling temperature')
    add_argument('vocab_path', dtype='path',
                 msg='Path to a vocabulary file which would provide word-level features to the  model.')
    add_argument('use_brute_force', dtype=bool, default=False, msg='Use brute force searcher.')
    add_argument('n_times', dtype=int, default=5, msg='Number of neighbors.')

    @not_supported_argument_value('new_style', True)
    def __init__(self):

        super().__init__()
        self.lm_model = LM()
        saved_dict = torch.load(g.lm_model_path)
        self.lm_model.load_state_dict(saved_dict['model'])
        freeze(self.lm_model)

        # NOTE(j_luo) I'm keeping a separate embedding for label prediction.
        self.emb_for_label = FeatEmbedding('feat_emb_for_label', 'chosen_feat_group', 'char_emb')

        cat_dim = g.dim * self.emb_for_label.effective_num_feature_groups
        self.self_attn_layers = nn.ModuleList()
        for _ in range(g.num_self_attn_layers):
            self.self_attn_layers.append(TransformerLayer(cat_dim, g.num_heads, cat_dim, dropout=g.dropout))
        self.positional_embedding = PositionalEmbedding(512, cat_dim)

        self.label_predictor = nn.Sequential(
            nn.Linear(cat_dim, cat_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(cat_dim, 3)  # BIO.
        )

        self.label_predictor[0].refine_names('weight', ['hid_repr', 'self_attn_repr'])
        self.label_predictor[2].refine_names('weight', ['label', 'hid_repr'])

        # Use vocab feature if provided.
        self.vocab = None
        if g.vocab_path:
            with open(g.vocab_path, 'r', encoding='utf8') as fin:
                self.vocab = set(line.strip() for line in fin)

        searcher_cls = BruteForceSearcher if g.use_brute_force else BeamSearcher
        self.searcher = searcher_cls()

        self.phi_scorer = nn.Linear(5, 1)
        self.phi_scorer.refine_names('weight', ['score', 'feature'])

    def _adapt(self, packed_feat_matrix: LT) -> LT:
        if g.adapt_mode == 'none':
            return packed_feat_matrix
        else:
            raise NotImplementedError()

    def forward(self, batch: Union[ContinuousIpaBatch, IpaBatch]) -> DecipherModelReturn:
        # Get the samples of label sequences first.
        out = self.emb_for_label(batch.feat_matrix, batch.source_padding)

        positions = get_named_range(batch.feat_matrix.size('length'), name='length')
        pos_emb = self.positional_embedding(positions).align_as(out)
        out = out + pos_emb
        out = out.align_to('length', 'batch', 'char_emb')
        with NoName(out, batch.source_padding):
            for i, layer in enumerate(self.self_attn_layers):
                out = layer(out, src_key_padding_mask=batch.source_padding)
        state = out.refine_names('length', 'batch', ...)
        logits = self.label_predictor(state)
        label_log_probs = logits.log_softmax(dim='label')
        label_probs = label_log_probs.exp()

        # NOTE(j_luo) O is equivalent to None.
        mask = expand_as(batch.source_padding, label_probs)
        source = expand_as(get_tensor([0.0, 0.0, 1.0]).refine_names('label').float(), label_probs)
        label_probs = label_probs.rename(None).masked_scatter(mask.rename(None), source.rename(None))
        label_probs = label_probs.refine_names('length', 'batch', 'label')

        if not self.training or (g.supervised and not g.train_phi):
            probs = DecipherModelProbReturn(label_log_probs, None)
            return DecipherModelReturn(state, probs, None, None, None, None, None)

        # ------------------ More info during training ----------------- #

        # Get the lm score.
        gold_tag_seqs = batch.gold_tag_seqs if g.supervised and g.train_phi else None
        samples, sample_log_probs = self.searcher.search(batch.lengths, label_log_probs, gold_tag_seqs=gold_tag_seqs)
        probs = DecipherModelProbReturn(label_log_probs, sample_log_probs)

        packed_words, scores = self._get_scores(samples,
                                                batch.segments,
                                                batch.lengths,
                                                batch.feat_matrix,
                                                batch.source_padding)

        if g.supervised and g.train_phi:
            return DecipherModelReturn(state, probs, packed_words, None, scores, None, None)

        # ------------------- Contrastive estimation ------------------- #

        ptb_segments = list()
        duplicates = list()
        for segment in batch.segments:
            _ptb_segments, _duplicates = segment.perturb_n_times(g.n_times)
            # NOTE(j_luo) Ignore the first one.
            ptb_segments.extend(_ptb_segments[1:])
            duplicates.extend(_duplicates[1:])
        # ptb_segments = [segment.perturb_n_times(5) for segment in batch.segments]
        ptb_feat_matrix = [segment.feat_matrix for segment in ptb_segments]
        ptb_feat_matrix = torch.nn.utils.rnn.pad_sequence(ptb_feat_matrix, batch_first=True)
        ptb_feat_matrix.rename_('batch', 'length', 'feat_group')
        samples = samples.align_to('batch', ...)
        with NoName(samples, batch.lengths, batch.source_padding):
            ptb_samples = torch.repeat_interleave(samples, g.n_times * 2, dim=0)
            ptb_lengths = torch.repeat_interleave(batch.lengths, g.n_times * 2, dim=0)
            ptb_source_padding = torch.repeat_interleave(batch.source_padding, g.n_times * 2, dim=0)
        ptb_samples.rename_(*samples.names)
        ptb_lengths.rename_('batch')
        ptb_source_padding.rename_('batch', 'length')

        ptb_packed_words, ptb_scores = self._get_scores(ptb_samples,
                                                        ptb_segments,
                                                        ptb_lengths,
                                                        ptb_feat_matrix,
                                                        ptb_source_padding)

        ret = DecipherModelReturn(state, probs, packed_words, ptb_packed_words, scores, ptb_scores, duplicates)
        return ret

    def _get_scores(self, samples: LT, segments: Sequence[SegmentWindow], lengths: LT, feat_matrix: LT, source_padding: BT) -> Tuple[PackedWords, DecipherModelScoreReturn]:
        bs = len(segments)

        segment_list = None
        if self.vocab is not None:
            segment_list = [segment.segment_list for segment in segments]
        packed_words = self.pack(samples, lengths,
                                 feat_matrix,
                                 segments,
                                 segment_list=segment_list)
        packed_words.word_feat_matrices = self._adapt(packed_words.word_feat_matrices)

        try:
            lm_batch = self._prepare_batch(packed_words)  # TODO(j_luo)  This is actually continous batching.
            scores = self._get_lm_scores(lm_batch)
            nlls = list()
            for cat, (nll, weight) in scores.items():
                if should_include(g.feat_groups, cat):
                    nlls.append(nll * weight)
            # nlls = sum(nlls)
            nlls = sum(nlls) / lm_batch.lengths
            bw = packed_words.word_lengths.size('batch_word')
            p = packed_words.word_positions.size('position')
            nlls = nlls.unflatten('batch', [('batch_word', bw), ('position', p)])
            nlls = nlls.sum(dim='position')
            lm_score, in_vocab_score = self._unpack(nlls, packed_words, bs)
        except EmptyPackedWords:
            lm_score = get_zeros(bs, packed_words.num_samples)
            in_vocab_score = get_zeros(bs, packed_words.num_samples)

        word_score = self._get_word_score(packed_words, bs)
        readable_score, unreadable_score = self._get_readable_scores(source_padding, samples)

        scores = [lm_score, word_score, in_vocab_score, readable_score, unreadable_score]
        features = torch.stack(scores, new_name='feature')
        phi_score = self.phi_scorer(features).squeeze('score')

        # if g.search:
        #     samples = samples.align_to('length', 'batch', 'sample')
        #     flat_samples = samples.flatten(['batch', 'sample'], 'batch_X_sample')
        #     flat_sample_embeddings = self.tag_embedding(flat_samples)
        #     bxs = flat_samples.size('batch_X_sample')
        #     h0 = get_zeros([1, bxs, 100])
        #     c0 = get_zeros([1, bxs, 100])
        #     with NoName(flat_sample_embeddings):
        #         output, (hn, _) = self.tag_lstm(flat_sample_embeddings, (h0, c0))
        #     tag_score = self.tag_scorer(hn).squeeze(dim=0).squeeze(dim=-1)
        #     tag_score = tag_score.view(samples.size('batch'), samples.size('sample'))
        #     ret['tag_score'] = tag_score.rename('batch', 'sample')
        scores = DecipherModelScoreReturn(lm_score, word_score, in_vocab_score,
                                          readable_score, unreadable_score,
                                          phi_score)

        return packed_words, scores

    @deprecated
    def _get_word_score(self, packed_words: PackedWords, batch_size: int) -> FT:
        with torch.no_grad():
            num_words = get_zeros(batch_size * packed_words.num_samples)
            bi = packed_words.batch_indices
            si = packed_words.sample_indices
            idx = (bi * packed_words.num_samples + si).rename(None)
            inc = get_zeros(packed_words.batch_indices.size('batch_word')).fill_(1.0)
            # TODO(j_luo) add scatter_add_ to named_tensor module
            num_words.scatter_add_(0, idx, inc)
            num_words = num_words.view(batch_size, packed_words.num_samples).refine_names('batch', 'sample')
        return num_words

    def _get_lm_scores(self, lm_batch: IpaBatch) -> Dict[Category, FT]:
        max_size = min(100000, lm_batch.batch_size)
        with torch.no_grad():
            batches = lm_batch.split(max_size)
            all_scores = [self.lm_model.score(batch) for batch in batches]
            cats = all_scores[0].keys()
            all_scores = {
                cat: list(zip(*[scores[cat] for scores in all_scores]))
                for cat in cats
            }
            for cat in cats:
                scores, weights = all_scores[cat]
                scores = torch.cat(scores, names='batch', new_name='batch')
                weights = torch.cat(weights, names='batch', new_name='batch')
                all_scores[cat] = (scores, weights)
        return all_scores

    @staticmethod
    def _prepare_batch(packed_words: PackedWords) -> IpaBatch:
        # TODO(j_luo) ugly
        try:
            return IpaBatch(
                None,
                packed_words.word_lengths.rename(None),
                packed_words.word_feat_matrices.rename(None),
                batch_name='batch',
                length_name='length'
            ).cuda()
        except RuntimeError:
            raise EmptyPackedWords()

    def pack(self, samples: LT, lengths: LT, feat_matrix: LT, segments: np.ndarray, segment_list: Optional[List[List[str]]] = None) -> PackedWords:
        with torch.no_grad():
            feat_matrix = feat_matrix.align_to('batch', 'length', 'feat_group')
            samples = samples.align_to('batch', 'sample', 'length').int()
            ns = samples.size('sample')
            lengths = lengths.align_to('batch', 'sample').expand(-1, ns).int()
            batch_indices, sample_indices, word_positions, word_lengths, is_unique = extract_words(
                samples.cpu().numpy(), lengths.cpu().numpy(), num_threads=4)

            in_vocab = np.zeros_like(batch_indices, dtype=np.bool)
            if self.vocab is not None:
                in_vocab = check_in_vocab(batch_indices, word_positions, word_lengths,
                                          segment_list, self.vocab, num_threads=4)
                in_vocab = get_tensor(in_vocab).refine_names('batch_word').bool()

            batch_indices = get_tensor(batch_indices).refine_names('batch_word').long()
            sample_indices = get_tensor(sample_indices).refine_names('batch_word').long()
            word_positions = get_tensor(word_positions).refine_names('batch_word', 'position').long()
            word_lengths = get_tensor(word_lengths).refine_names('batch_word').long()
            is_unique = get_tensor(is_unique).refine_names('batch', 'sample').bool()

            key = (
                batch_indices.align_as(word_positions).rename(None),
                word_positions.rename(None)
            )
            word_feat_matrices = feat_matrix.rename(None)[key]
            word_feat_matrices = word_feat_matrices.refine_names('batch_word', 'position', 'feat_group')
            packed_words = PackedWords(word_feat_matrices,
                                       word_lengths,
                                       batch_indices,
                                       sample_indices,
                                       word_positions,
                                       is_unique,
                                       ns,
                                       segments,
                                       in_vocab=in_vocab)
            return packed_words

    def _unpack(self, nlls: FT, packed_words: PackedWords, batch_size: int) -> Tuple[FT, FT]:
        with torch.no_grad():
            lm_loss = get_zeros(batch_size * packed_words.num_samples)
            bi = packed_words.batch_indices
            si = packed_words.sample_indices
            idx = (bi * packed_words.num_samples + si).rename(None)
            # TODO(j_luo) ugly
            lm_loss.scatter_add_(0, idx, nlls.rename(None))
            lm_loss = lm_loss.view(batch_size, packed_words.num_samples).refine_names('batch', 'sample')

            in_vocab_score = get_zeros(batch_size * packed_words.num_samples)
            if self.vocab is not None:
                in_vocab_score.scatter_add_(0, idx, packed_words.in_vocab.float().rename(None))
                in_vocab_score = in_vocab_score.view(
                    batch_size, packed_words.num_samples).refine_names('batch', 'sample')

        return -lm_loss, in_vocab_score  # NOTE(j_luo) NLL are losses, not scores.

    @deprecated
    def _sample(self, label_probs: FT, sampling_probs: FT, source_padding: FT, gold_tag_seqs: Optional[FT] = None) -> Tuple[LT, FT]:
        """Return samples based on `label_probs`."""
        # Ignore padded indices.
        label_probs = label_probs.align_to('batch', 'length', 'label')
        sampling_probs = sampling_probs.align_to('batch', 'length', 'label')
        source_padding = source_padding.align_to('batch', 'length')

        # Get packed batches.
        label_distr = Categorical(probs=sampling_probs.rename(None))
        label_samples = label_distr.sample([g.num_samples]).refine_names('sample', 'batch', 'length')
        label_samples = label_samples.align_to('batch', 'sample', 'length')
        # Add the ground truth if needed.
        if gold_tag_seqs is not None:
            gold_tag_seqs = gold_tag_seqs.align_as(label_samples)
            all_other_tag_seqs = torch.full_like(gold_tag_seqs, O)
            label_samples = torch.cat([gold_tag_seqs, all_other_tag_seqs, label_samples], dim='sample')
        batch_idx = get_named_range(label_samples.size('batch'), 'batch').align_as(label_samples).rename(None)
        length_idx = get_named_range(label_samples.size('length'), 'length').align_as(label_samples).rename(None)
        label_sample_probs = label_probs.rename(None)[batch_idx, length_idx, label_samples.rename(None)]
        label_sample_probs = label_sample_probs.refine_names(*label_samples.names)
        label_sample_log_probs = (1e-8 + label_sample_probs).log()
        label_sample_log_probs = ((~source_padding).align_as(label_sample_log_probs).float()
                                  * label_sample_log_probs).sum(dim='length')
        return label_samples, label_sample_log_probs

    def _get_readable_scores(self, source_padding: BT, samples: LT) -> Tuple[FT, FT]:
        samples = samples.align_to('batch', 'sample', 'length')
        source_padding = source_padding.align_as(samples)
        is_part_of_word = ((samples == B) | (samples == I)) & ~source_padding
        not_part_of_word = (samples == O) & ~source_padding
        readable_score = is_part_of_word.float().sum(dim='length')
        unreadable_score = not_part_of_word.float().sum(dim='length')
        return readable_score, unreadable_score
