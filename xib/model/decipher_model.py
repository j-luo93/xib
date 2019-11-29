import math
from collections import defaultdict
from dataclasses import InitVar, dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.modules import MultiheadAttention
from torch.nn.modules.transformer import TransformerEncoderLayer

from dev_misc.arglib import add_argument, g, not_supported_argument_value
from dev_misc.devlib import (dataclass_numpy, dataclass_size_repr, get_array,
                             get_tensor, get_zeros)
from dev_misc.devlib.named_tensor import (NoName, expand_as, get_named_range,
                                          self_attend)
from dev_misc.trainlib import freeze
from dev_misc.utils import cached_property, deprecated
from xib.check_in_vocab_impl import \
    check_in_vocab  # pylint: disable=no-name-in-module
from xib.data_loader import ContinuousTextIpaBatch, IpaBatch
from xib.extract_words_impl import extract_words_v8 as extract_words  # pylint: disable=no-name-in-module
from xib.gumbel import gumbel_softmax
from xib.ipa import Category, should_include
from xib.ipa.process import B, I, O, Segmentation, Span
from xib.model.modules import Predictor

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
        new_names = positions.names + ('char_emb', )
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


@deprecated
class OldDecipherModel(nn.Module):

    add_argument('adapt_mode', default='none', choices=['none'], dtype=str,
                 msg='how to adapt the features from one language to another')
    add_argument('num_self_attn_layers', default=2, dtype=int, msg='number of self attention layers')
    add_argument('num_samples', default=100, dtype=int, msg='number of samples per sequence')
    add_argument('num_heads', default=4, dtype=int, msg='Number for heads for self attention.')
    add_argument('lm_model_path', dtype='path', msg='path to a pretrained lm model')
    add_argument('dropout', default=0.2, dtype=float, msg='dropout rate')
    add_argument('use_local_probs', default=True, dtype=bool, msg='Flag to use local probs for producing global probs.')
    add_argument('sampling_temperature', default=1.0, dtype=float, msg='Sampling temperature')
    add_argument('vocab_path', dtype='path',
                 msg='Path to a vocabulary file which would provide word-level features to the  model.')

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

        # For supervised mode, you want to use a global predictor to predict the probs for sequences.
        seq_scorer_dim = 2 + (self.vocab is not None) + (g.use_local_probs)
        # if g.supervised:
        self.seq_scorer = nn.Sequential(
            nn.Linear(seq_scorer_dim, 1),
        )
        self.seq_scorer[0].refine_names('weight', ['score', 'seq_feat'])

        if g.use_mlm_loss:
            self.predictor = Predictor(cat_dim)

    def _adapt(self, packed_feat_matrix: LT) -> LT:
        if g.adapt_mode == 'none':
            return packed_feat_matrix
        else:
            raise NotImplementedError()

    def forward(self, batch: Union[ContinuousTextIpaBatch, IpaBatch], mode: str):
        bs = batch.batch_size
        ret = dict()

        # ------------------------- Local mode ------------------------- #

        # Get the samples of label sequences first.
        # HACK(j_luo)
        masked_positions = None
        if isinstance(batch, IpaBatch):
            masked_positions = batch.pos_to_predict
        out = self.emb_for_label(batch.feat_matrix, batch.source_padding, masked_positions=masked_positions)

        positions = get_named_range(batch.feat_matrix.size('length'), name='length')
        pos_emb = self.positional_embedding(positions).align_as(out)
        out = out + pos_emb
        out = out.align_to('length', 'batch', 'char_emb')
        for i, layer in enumerate(self.self_attn_layers):
            # out, _ = self_attend(layer, out, f'self_attn_repr')
            with NoName(out, batch.source_padding):
                out = layer(out, src_key_padding_mask=batch.source_padding)
        out = out.refine_names('length', 'batch', None)
        ret['out'] = out
        logits = self.label_predictor(out)
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

        # HACK(j_luo)
        temperature = g.sampling_temperature if self.training else 0.1
        sampling_probs = (logits / temperature).log_softmax(dim='label').exp()
        samples, sample_log_probs = self._sample(
            label_probs, sampling_probs, batch.source_padding, gold_tag_seqs=gold_tag_seqs)

        # Get the lm score.
        # TODO(j_luo) unique is still a bit buggy -- BIO is the same as IIO.
        segment_list = None
        if self.vocab is not None:
            segment_list = [segment.segment_list for segment in batch.segments]
        packed_words, is_unique = self.pack(samples, batch.lengths, batch.feat_matrix,
                                            batch.segments, segment_list=segment_list)
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
        lm_score, in_vocab_score = self._unpack(nlls, packed_words, bs)

        # DEBUG(j_luo)
        # lm_score = lm_score.exp()

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

        if self.vocab is not None:
            ret['in_vocab_score'] = in_vocab_score

        # if g.supervised:
            # features = torch.stack([ret['lm_score'], ret['word_score']], new_name='seq_feat')
        word_level_features = [lm_score, word_score]
        if self.vocab is not None:
            word_level_features.append(in_vocab_score)
        if g.use_local_probs:
            word_level_features.append(sample_log_probs.exp())
        features = torch.stack(word_level_features, new_name='seq_feat')
        # features = torch.stack([ret['sample_log_probs'].exp(), ret['lm_score'],
        #                         ret['word_score']], new_name='seq_feat')
        seq_scores = self.seq_scorer(features).squeeze(dim='score')
        # DEBUG(j_luo) # HACK(j_luo)
        ret['seq_scores'] = seq_scores.exp() / 5.0

        modified_seq_scores = seq_scores + (~is_unique).float() * (-999.9)
        seq_log_probs = modified_seq_scores.log_softmax(dim='sample')
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

    def pack(self, samples: LT, lengths: LT, feat_matrix: LT, segments: np.ndarray, segment_list: Optional[List[List[str]]] = None) -> Tuple[PackedWords, BT]:
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

            # TODO(j_luo) ugly
            key = (
                batch_indices.align_as(word_positions).rename(None),
                word_positions.rename(None)
            )
            word_feat_matrices = feat_matrix.rename(None)[key]
            word_feat_matrices = word_feat_matrices.refine_names('batch_word', 'position', 'feat_group')
            packed_words = PackedWords(word_feat_matrices, word_lengths, batch_indices, sample_indices, word_positions,
                                       is_unique,
                                       ns,
                                       segments,
                                       in_vocab=in_vocab)
            return packed_words, is_unique

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


class DecipherModel(OldDecipherModel):

    add_argument('unreadable_baseline', dtype=float, default=0.001,
                 msg='Baseline probability for unextracted characters.')
    add_argument('gumbel_vae', dtype=bool, default=False, msg='Use gumbel VAE loss.')

    def forward(self, batch: Union[ContinuousTextIpaBatch, IpaBatch], mode: str):
        assert mode == 'risk'  # HACK(j_luo)
        bs = batch.batch_size
        ret = dict()

        # Get the samples of label sequences first.
        out = self.emb_for_label(batch.feat_matrix, batch.source_padding)
        positions = get_named_range(batch.feat_matrix.size('length'), name='length')
        pos_emb = self.positional_embedding(positions).align_as(out)
        out = out + pos_emb
        out = out.align_to('length', 'batch', 'char_emb')
        for i, layer in enumerate(self.self_attn_layers):
            # out, _ = self_attend(layer, out, f'self_attn_repr')
            with NoName(out, batch.source_padding):
                out = layer(out, src_key_padding_mask=batch.source_padding)
        out = out.refine_names('length', 'batch', None)
        ret['out'] = out
        logits = self.label_predictor(out)
        if g.gumbel_vae:
            # DEBUG(j_luo)
            try:
                self._cnt += 1
            except:
                self._cnt = 1
                self._temp = 10.0
            if self._cnt % 10 == 0:
                self._temp *= 0.95
                self._temp = max(self._temp, 1.0)
                print(self._temp)

            label_probs, label_probs_hard, samples = gumbel_softmax(logits, self._temp)  # , g.num_samples)

            # DEBUG(j_luo)
            # from itertools import product
            # samples = get_tensor(list(product([B, I, O], repeat=3))).refine_names('sample', 'length')

            _, _, best_samples = gumbel_softmax(logits, 0.1)  # , 1)  # DEBUG(j_luo)
            label_log_probs = (1e-8 + label_probs).log()
            label_log_probs_hard = (label_probs_hard + 1e-8).log()
            sample_log_probs = None
        elif g.search:
            search_result = self.searcher.search(batch)
            rs = search_result['readable_score']
            urs = search_result['unreadable_score']
            lms = search_result['lm_score']
            ivs = search_result['in_vocab_score']
            ds = search_result['diff_score']
            fv = torch.stack([rs, urs, lms, ivs, ds], new_name='feature')
            return {'feature': fv}
        else:
            # DEBUG(j_luo)
            label_probs, _, _ = gumbel_softmax(logits, 1.0)  # , g.num_samples)
            label_log_probs = (1e-8 + label_probs).log()

            # label_log_probs = logits.log_softmax(dim='label')
            # label_probs = label_log_probs.exp()

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

            # DEBUG(j_luo)
            # temperature = g.sampling_temperature if self.training else 0.1
            # sampling_probs = (logits / temperature).log_softmax(dim='label').exp()
            sampling_probs = label_probs

            samples, sample_log_probs = self._sample(
                label_probs, sampling_probs, batch.source_padding, gold_tag_seqs=gold_tag_seqs)

        ret['label_probs'] = label_probs
        ret['label_log_probs'] = label_log_probs
        ret['sample_log_probs'] = sample_log_probs

        scores = self.get_scores(samples, batch)
        # DEBUG(j_luo)
        # from pprint import pprint
        # pprint(packed_words.sampled_segments_by_batch[0])

        # try:
        #     self._cnt += 1
        # except:
        #     self._cnt = 1
        # if self._cnt % 300 == 0:
        #     breakpoint()  # DEBUG(j_luo)

        if g.gumbel_vae:

            # if self._cnt % 300 == 0:
            #     breakpoint()  # DEBUG(j_luo)

            # total_score = lm_score + unreadable_score + in_vocab_score
            sample_score = ret['sample_score']

            weight = (~batch.source_padding).float()
            sample_log_probs = label_log_probs_hard.gather('label', samples)
            _log_probs = sample_log_probs * weight.align_as(sample_log_probs)
            _log_probs = _log_probs.sum(dim='length')
            q_score = _log_probs.exp().align_as(sample_score) * sample_score
            # q_score = (q_score + unreadable_score + in_vocab_score + readable_score).sum(dim='sample')
            # import time; time.sleep(0.5)
            q_score = q_score.sum(dim='sample')  # / g.num_samples
            # lm_score = (lm_score.align_as(label_log_probs_hard) * label_log_probs_hard)
            # lm_score = (lm_score + ret['unreadable_score']) * weight.align_as(lm_score)
            # lm_score = lm_score.sum(dim='label').sum(dim='sample')
            # DEBUG(j_luo) No kl so far
            # kl_tmp = label_probs * (label_log_probs - math.log(1.0 / 3))
            # kl = (kl_tmp * weight.align_as(kl_tmp)).sum(dim='label').sum(dim='length')
            elbo = q_score  # - kl
            ret['elbo'] = elbo

            # DEBUG(j_luo)
            segment_list = None
            if self.vocab is not None:
                segment_list = [segment.segment_list for segment in batch.segments]
            best_packed_words, _ = self.pack(best_samples, batch.lengths, batch.feat_matrix,
                                             batch.segments, segment_list=segment_list)

            # DEBUG(j_luo)
            # print('-' * 30)
            # print('logits:')
            # print(logits)
            # print('label_probs:')
            # print(label_probs)
            # print('sampled:')
            # print(packed_words.sampled_segments_by_batch)
            # print('best:')
            # print(best_packed_words.sampled_segments_by_batch)
            # print('elbo:')
            # print(elbo)
            # print('q_score:')
            # print(q_score.sum(), q_score.sum() / readable_score.sum())
            # print('unreadable:')
            # print(unreadable_score.sum())

            # import time; time.sleep(0.5) # DEBUG(j_luo)

        return ret

    def get_scores(self, samples: LT, batch: ContinuousTextIpaBatch):
        ret = dict()
        bs = batch.batch_size
        # Get the lm score.
        # TODO(j_luo) unique is still a bit buggy -- BIO is the same as IIO.
        segment_list = None
        if self.vocab is not None:
            segment_list = [segment.segment_list for segment in batch.segments]
        packed_words, is_unique = self.pack(samples, batch.lengths, batch.feat_matrix,
                                            batch.segments, segment_list=segment_list)
        packed_words.word_feat_matrices = self._adapt(packed_words.word_feat_matrices)
        try:
            lm_batch = self._prepare_batch(packed_words)  # TODO(j_luo)  This is actually continous batching.
            scores = self._get_lm_scores(lm_batch)
            nlls = list()
            for cat, (nll, weight) in scores.items():
                if should_include(g.feat_groups, cat):
                    nlls.append(nll * weight)
            # NOTE(j_luo) Do not normalize the scores by length.
            nlls = sum(nlls) / torch.pow(lm_batch.lengths.float(), 0.5)
            # nlls = sum(nlls)
            bw = packed_words.word_lengths.size('batch_word')
            p = packed_words.word_positions.size('position')
            nlls = nlls.unflatten('batch', [('batch_word', bw), ('position', p)])
            nlls = nlls.sum(dim='position')
            lm_score, in_vocab_score = self._unpack(nlls, packed_words, bs)
        except EmptyPackedWords:
            lm_score = get_zeros(1, 1).rename('batch', 'sample')
            in_vocab_score = get_zeros(1, 1).rename('batch', 'sample')

        # Compute word score that corresponds to the number of readable words.
        readable_score, unreadable_score = self._get_readable_scores(batch.source_padding, samples)
        ret['readable_score'] = readable_score
        ret['unreadable_score'] = unreadable_score

        ret.update(
            readable_score=readable_score,
            lm_score=lm_score,
            is_unique=is_unique,
        )

        if self.vocab is not None:
            ret['in_vocab_score'] = in_vocab_score

        # modified_seq_scores = seq_scores + (~is_unique).float() * (-999.9)
        # seq_log_probs = modified_seq_scores.log_softmax(dim='sample')
        # ret['seq_log_probs'] = seq_log_probs
        # DEBUG(j_luo)
        ret['packed_words'] = packed_words
        word_score = self._get_word_score(packed_words, bs)
        diff_score = (word_score - in_vocab_score) * (-5.0)
        sample_score = ret['readable_score'] + ret['unreadable_score'] + \
            ret['lm_score'] + ret['in_vocab_score'] + diff_score
        ret['sample_score'] = sample_score
        ret['diff_score'] = diff_score
        return ret

    def _get_readable_scores(self, source_padding: BT, samples: LT) -> Tuple[FT, FT]:
        samples = samples.align_to('batch', 'sample', 'length')
        source_padding = source_padding.align_as(samples)
        is_part_of_word = ((samples == B) | (samples == I)) & ~source_padding
        not_part_of_word = (samples == O) & ~source_padding
        readable_score = is_part_of_word.float().sum(dim='length')
        unreadable_score = not_part_of_word.float().sum(dim='length') * math.log(g.unreadable_baseline)
        return readable_score, unreadable_score


class TransferModel(DecipherModel):

    def __init__(self):
        super().__init__()

        self.adapter = AdaptLayer()

    def forward(self, batch: ContinuousTextIpaBatch, mode: str):
        ret = super().forward(batch, mode)
        ...  # FIXME(j_luo) fill in the adapter part
        return ret
