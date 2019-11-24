from torch.nn.modules.transformer import TransformerEncoderLayer
from collections import defaultdict
from dataclasses import InitVar, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.modules import MultiheadAttention

from dev_misc.arglib import (add_argument, g,
                             not_supported_argument_value)
from dev_misc.devlib import (dataclass_numpy, dataclass_size_repr, get_array,
                             get_tensor, get_zeros)
from dev_misc.devlib.named_tensor import (NoName, expand_as, get_named_range,
                                          self_attend)
from dev_misc.trainlib import freeze
from dev_misc.utils import cached_property
from xib.data_loader import ContinuousTextIpaBatch, IpaBatch
from xib.extract_words_impl import extract_words_v8 as extract_words  # pylint: disable=no-name-in-module
from xib.ipa import Category, should_include
from xib.ipa.process import Segmentation, Span

from . import BT, FT, LT
from .lm_model import LM
from .modules import FeatEmbedding


@dataclass
class PackedWords:
    word_feat_matrices: LT
    word_lengths: LT
    batch_indices: LT
    sample_indices: LT
    word_positions: LT
    num_samples: int
    orig_segments: np.ndarray

    __repr__ = dataclass_size_repr
    numpy = dataclass_numpy

    @property
    def batch_size(self):
        return len(self.orig_segments)

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


class PositionalEmbedding(nn.Module):

    def __init__(self, n_pos, dim):
        super().__init__()
        embeddings = torch.zeros(n_pos, dim)
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ])
        embeddings[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        embeddings[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        self.register_buffer('embeddings', embeddings)

    def forward(self, positions: LT):
        with NoName(self.embeddings, positions):
            ret = self.embeddings[positions]
        new_names = positions.names + ('char_emb_for_label', )
        return ret.refine_names(*new_names)


class SelfAttention(MultiheadAttention):
    """Always set `_qkv_same_embed_dim` to False."""

    @property
    def _qkv_same_embed_dim(self):
        return False

    @_qkv_same_embed_dim.setter
    def _qkv_same_embed_dim(self, value):
        pass


class TransformerLayer(TransformerEncoderLayer):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.self_attn = SelfAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.nn.functional.gelu


class DecipherModel(nn.Module):

    add_argument('adapt_mode', default='none', choices=['none'], dtype=str,
                 msg='how to adapt the features from one language to another')
    add_argument('num_self_attn_layers', default=2, dtype=int, msg='number of self attention layers')
    add_argument('num_samples', default=100, dtype=int, msg='number of samples per sequence')
    add_argument('num_heads', default=4, dtype=int, msg='Number for heads for self attention.')
    add_argument('lm_model_path', dtype='path', msg='path to a pretrained lm model')
    add_argument('dropout', default=0.2, dtype=float, msg='dropout rate')

    @not_supported_argument_value('new_style', True)
    def __init__(self):

        super().__init__()
        self.lm_model = LM()
        saved_dict = torch.load(g.lm_model_path)
        self.lm_model.load_state_dict(saved_dict['model'])
        freeze(self.lm_model)

        # NOTE(j_luo) I'm keeping a separate embedding for label prediction.
        self.emb_for_label = FeatEmbedding('feat_emb_for_label', 'chosen_feat_group', 'char_emb_for_label')

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

        # For supervised mode, you want to use a global predictor to predict the probs for sequences.
        if g.supervised:
            self.seq_scorer = nn.Sequential(
                nn.Linear(3, 1),
            )
            self.seq_scorer[0].refine_names('weight', ['score', 'seq_feat'])

    def _adapt(self, packed_feat_matrix: LT) -> LT:
        if g.adapt_mode == 'none':
            return packed_feat_matrix
        else:
            raise NotImplementedError()

    def forward(self, batch: ContinuousTextIpaBatch, mode: str):
        bs = batch.batch_size
        ret = dict()

        # ------------------------- Local mode ------------------------- #

        # Get the samples of label sequences first.
        out = self.emb_for_label(batch.feat_matrix, batch.source_padding)
        positions = get_named_range(batch.feat_matrix.size('length'), name='length')
        pos_emb = self.positional_embedding(positions)
        out = out + pos_emb
        out = out.align_to('length', 'batch', 'char_emb_for_label')
        for i, layer in enumerate(self.self_attn_layers):
            # out, _ = self_attend(layer, out, f'self_attn_repr')
            with NoName(out, batch.source_padding):
                out = layer(out, src_key_padding_mask=batch.source_padding)
        out = out.refine_names('length', 'batch', None)
        logits = self.label_predictor(out)  # * 0.01  # HACK(j_luo) use 0.01 to make it smooth
        label_log_probs = logits.log_softmax(dim='label')
        label_probs = label_log_probs.exp()
        ret['label_probs'] = label_probs
        ret['label_log_probs'] = label_log_probs
        if mode == 'local':
            return ret

        # ------------------------- Global mode ------------------------ #

        # NOTE(j_luo) O is equivalent to None.
        mask = expand_as(batch.source_padding, label_probs)
        source = expand_as(get_tensor([0.0, 0.0, 1.0]).refine_names('label').float(), label_probs)
        # TODO(j_luo) ugly
        label_probs = label_probs.rename(None).masked_scatter(mask.rename(None), source.rename(None))
        label_probs = label_probs.refine_names('length', 'batch', 'label')

        if g.supervised and self.training:
            gold_tag_seqs = batch.gold_tag_seqs
            if gold_tag_seqs is None:
                raise RuntimeError(f'Gold tag seqsuence must be provided in supervised mode.')
        else:
            gold_tag_seqs = None
        samples, sample_log_probs = self._sample(label_probs, batch.source_padding, gold_tag_seqs=gold_tag_seqs)

        # Get the lm score.
        # TODO(j_luo) unique is still a bit buggy -- BIO is the same as IIO.
        packed_words, is_unique = self.pack(samples, batch.lengths, batch.feat_matrix, batch.segments)
        packed_words.word_feat_matrices = self._adapt(packed_words.word_feat_matrices)
        lm_batch = self._prepare_batch(packed_words)  # TODO(j_luo)  This is actually continous batching.
        scores = self._get_lm_scores(lm_batch)
        nlls = list()
        for cat, (nll, weight) in scores.items():
            if should_include(g.feat_groups, cat):
                nlls.append(nll * weight)
        nlls = sum(nlls) / lm_batch.lengths  # NOTE(j_luo) Average NLL by word length
        bw = packed_words.word_lengths.size('batch_word')
        p = packed_words.word_positions.size('position')
        nlls = nlls.unflatten('batch', [('batch_word', bw), ('position', p)])
        nlls = nlls.sum(dim='position')
        lm_score = self._unpack(nlls, packed_words, bs)

        # DEBUG(j_luo)
        # print(packed_words.sampled_segments)
        # print(packed_words.sampled_segments_by_batch)
        # print(packed_words.get_segment_nlls(nlls))

        # Compute word score that corresponds to the number of readable words.
        word_score = self._get_word_score(packed_words, bs)

        ret.update(
            word_score=word_score,
            lm_score=lm_score,
            sample_log_probs=sample_log_probs,
            is_unique=is_unique,
        )
        if g.supervised:
            # features = torch.stack([ret['lm_score'], ret['word_score']], new_name='seq_feat')
            features = torch.stack([ret['sample_log_probs'].exp(), ret['lm_score'],
                                    ret['word_score']], new_name='seq_feat')
            seq_scores = self.seq_scorer(features).squeeze(dim='score')
            ret['seq_scores'] = seq_scores
            modified_seq_log_probs = seq_scores + (~is_unique).float() * (-999.9)
            seq_log_probs = modified_seq_log_probs.log_softmax(dim='sample')
            ret['seq_log_probs'] = seq_log_probs
            ret['packed_words'] = packed_words
        return ret

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
        max_size = min(300000, lm_batch.batch_size)
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
        return IpaBatch(
            None,
            packed_words.word_lengths.rename(None),
            packed_words.word_feat_matrices.rename(None),
            batch_name='batch',
            length_name='length'
        ).cuda()

    def pack(self, samples: LT, lengths: LT, feat_matrix: LT, segments: np.ndarray) -> Tuple[PackedWords, BT]:
        with torch.no_grad():
            feat_matrix = feat_matrix.align_to('batch', 'length', 'feat_group')
            samples = samples.align_to('batch', 'sample', 'length').int()
            ns = samples.size('sample')
            lengths = lengths.align_to('batch', 'sample').expand(-1, ns).int()
            batch_indices, sample_indices, word_positions, word_lengths, is_unique = extract_words(
                samples.cpu().numpy(), lengths.cpu().numpy(), num_threads=4)
            batch_indices = get_tensor(batch_indices).refine_names('batch_word').long()
            sample_indices = get_tensor(sample_indices).refine_names('batch_word').long()
            word_positions = get_tensor(word_positions).refine_names('batch_word', 'position').long()
            word_lengths = get_tensor(word_lengths).refine_names('batch_word').long()
            is_unique = get_tensor(is_unique).refine_names('batch', 'sample').bool()

            # TODO(j_luo) ugly
            key = (
                batch_indices.align_as(word_positions).rename(None),
                word_positions.rename(None)
            )
            word_feat_matrices = feat_matrix.rename(None)[key]
            word_feat_matrices = word_feat_matrices.refine_names('batch_word', 'position', 'feat_group')
            packed_words = PackedWords(word_feat_matrices, word_lengths, batch_indices, sample_indices, word_positions,
                                       ns,
                                       segments)
            return packed_words, is_unique

    def _unpack(self, nlls: FT, packed_words: PackedWords, batch_size: int) -> FT:
        with torch.no_grad():
            ret = get_zeros(batch_size * packed_words.num_samples)
            bi = packed_words.batch_indices
            si = packed_words.sample_indices
            idx = (bi * packed_words.num_samples + si).rename(None)
            # TODO(j_luo) ugly
            ret.scatter_add_(0, idx, nlls.rename(None))
            ret = ret.view(batch_size, packed_words.num_samples).refine_names('batch', 'sample')
        return -ret  # NOTE(j_luo) NLL are losses, not scores.

    def _sample(self, label_probs: FT, source_padding: FT, gold_tag_seqs: Optional[FT] = None) -> Tuple[LT, FT]:
        """Return samples based on `label_probs`."""
        # Ignore padded indices.
        label_probs = label_probs.align_to('batch', 'length', 'label')
        source_padding = source_padding.align_to('batch', 'length')

        # Get packed batches.
        label_distr = Categorical(probs=label_probs.rename(None))
        label_samples = label_distr.sample([g.num_samples]).refine_names('sample', 'batch', 'length')
        label_samples = label_samples.align_to('batch', 'sample', 'length')
        # Add the ground truth if needed.
        if gold_tag_seqs is not None:
            gold_tag_seqs = gold_tag_seqs.align_as(label_samples)
            label_samples = torch.cat([gold_tag_seqs, label_samples], dim='sample')
        batch_idx = get_named_range(label_samples.size('batch'), 'batch').align_as(label_samples).rename(None)
        length_idx = get_named_range(label_samples.size('length'), 'length').align_as(label_samples).rename(None)
        label_sample_probs = label_probs.rename(None)[batch_idx, length_idx, label_samples.rename(None)]
        label_sample_probs = label_sample_probs.refine_names(*label_samples.names)
        label_sample_log_probs = (1e-8 + label_sample_probs).log()
        label_sample_log_probs = ((~source_padding).align_as(label_sample_log_probs).float()
                                  * label_sample_log_probs).sum(dim='length')
        return label_samples, label_sample_log_probs
