from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.modules import MultiheadAttention

from arglib import add_argument, g, init_g_attr, not_supported_argument_value
from devlib import dataclass_numpy, dataclass_size_repr, get_tensor, get_zeros
from devlib.named_tensor import expand_as, get_named_range, self_attend
from xib.data_loader import ContinuousTextIpaBatch, IpaBatch
from xib.extract_words_impl import extract_words_v7 as extract_words
from xib.ipa import Category, should_include

from . import FT, LT, BT
from .lm_model import LM
from .modules import FeatEmbedding


@dataclass
class PackedWords:
    word_feat_matrices: LT
    word_lengths: LT
    batch_indices: LT
    sample_indices: LT
    word_positions: LT

    __repr__ = dataclass_size_repr
    numpy = dataclass_numpy


@init_g_attr(default='property')
class DecipherModel(nn.Module):

    add_argument('adapt_mode', default='none', choices=['none'], dtype=str,
                 msg='how to adapt the features from one language to another')
    add_argument('num_self_attn_layers', default=2, dtype=int, msg='number of self attention layers')
    add_argument('num_samples', default=100, dtype=int, msg='number of samples per sequence')
    add_argument('lm_model_path', dtype='path', msg='path to a pretrained lm model')

    @not_supported_argument_value('new_style', True)
    def __init__(self,
                 lm_model_path,
                 num_features,
                 dim,
                 adapt_mode,
                 num_self_attn_layers,
                 feat_groups,
                 num_samples):

        super().__init__()
        self.lm_model = LM()
        saved_dict = torch.load(lm_model_path)
        self.lm_model.load_state_dict(saved_dict['model'])

        # NOTE(j_luo) I'm keeping a separate embedding for label prediction.
        self.emb_for_label = FeatEmbedding('feat_emb_for_label', 'chosen_feat_group', 'char_emb_for_label')

        cat_dim = dim * self.emb_for_label.effective_num_feature_groups
        self.self_attn_layers = nn.ModuleList()
        for _ in range(num_self_attn_layers):
            self.self_attn_layers.append(MultiheadAttention(cat_dim, 4))

        self.label_predictor = nn.Sequential(
            nn.Linear(cat_dim, cat_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(cat_dim, 3)  # BIO.
        )

    def _adapt(self, packed_feat_matrix: LT) -> LT:
        if self.adapt_mode == 'none':
            return packed_feat_matrix
        else:
            raise NotImplementedError()

    def forward(self, batch: ContinuousTextIpaBatch):
        bs = batch.batch_size
        # Get the samples of label sequences first.
        out = self.emb_for_label(batch.feat_matrix, batch.source_padding)
        out = out.align_to('length', 'batch', 'char_emb_for_label')
        for i, layer in enumerate(self.self_attn_layers):
            out, _ = self_attend(layer, out, f'self_attn_repr')
        logits = self.label_predictor(out.rename(None)).refine_names(*out.names)
        logits = logits.rename(**{f'self_attn_repr': 'label'})
        label_probs = logits.log_softmax(dim=-1).exp()
        samples, sample_log_probs = self._sample(label_probs, batch.source_padding)

        # Get the lm score.
        # TODO(j_luo) unique is still a bit buggy -- BIO is the same as IIO.
        packed_words, is_unique = self._pack(samples, batch.lengths, batch.feat_matrix)
        # DEBUG(j_luo)
        segments = [segment.split('-') for segment in batch.segments]
        pw = packed_words.numpy()
        from collections import defaultdict
        all_tmp = defaultdict(lambda: defaultdict(list))
        for bi, si, wps, wl in zip(pw.batch_indices, pw.sample_indices, pw.word_positions, pw.word_lengths):
            tmp = [segments[bi][wp] for i, wp in enumerate(wps) if i < wl]
            all_tmp[bi][si].append(tmp)
        for bi in all_tmp:
            for si in range(self.num_samples):
                print(''.join(segments[bi]), si, end='')
                if si in all_tmp[bi]:
                    tmp = [''.join(t) for t in all_tmp[bi][si]]
                else:
                    tmp = list()
                print('', tmp)
        # breakpoint()  # DEBUG(j_luo)

        packed_words.word_feat_matrices = self._adapt(packed_words.word_feat_matrices)
        lm_batch = self._prepare_batch(packed_words)  # FIXME(j_luo)  This is actually continous batching.
        scores = self._get_lm_scores(lm_batch)
        nlls = list()
        for cat, (nll, weight) in scores.items():
            if should_include(self.feat_groups, cat):
                nlls.append(nll * weight)
        nlls = sum(nlls)
        lm_score = self._unpack(nlls, packed_words, bs)

        # Compute word score that corresponds to the number of readable words.
        word_score = self._get_word_score(packed_words, bs)

        return {
            'word_score': word_score,
            'lm_score': lm_score,
            'sample_log_probs': sample_log_probs,
            'is_unique': is_unique
        }

    def _get_word_score(self, packed_words: PackedWords, batch_size: int) -> FT:
        with torch.no_grad():
            num_words = get_zeros(batch_size * self.num_samples)
            bi = packed_words.batch_indices
            si = packed_words.sample_indices
            idx = (bi * self.num_samples + si).rename(None)
            inc = get_zeros(packed_words.batch_indices.size('batch_word')).fill_(1.0)
            num_words.scatter_add_(0, idx, inc)
            num_words = num_words.view(batch_size, self.num_samples).refine_names('batch', 'sample')
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
                names = scores[0].names
                scores = torch.cat([score.rename(None) for score in scores], dim=0).refine_names(*names)
                weights = torch.cat([weight.rename(None) for weight in weights], dim=0).refine_names(*names)
                all_scores[cat] = (scores, weights)
        return all_scores

    @staticmethod
    def _prepare_batch(packed_words: PackedWords) -> IpaBatch:
        # TODO(j_luo) Why does add_module (methods from Tensor) show up in the list autocompletion in PackedWords.
        # TODO(j_luo) ugly
        return IpaBatch(
            None,
            packed_words.word_lengths.rename(None),
            packed_words.word_feat_matrices.rename(None),
            batch_name='batch',
            length_name='length'
        ).cuda()

    def _pack(self, samples: LT, lengths: LT, feat_matrix: LT) -> Tuple[PackedWords, BT]:
        # TODO(j_luo) ugly
        with torch.no_grad():
            feat_matrix = feat_matrix.align_to('batch', 'length', 'feat_group')
            samples = samples.align_to('batch', 'sample', 'length').int()
            lengths = lengths.align_to('batch', 'sample').expand(-1, self.num_samples).int()
            batch_indices, sample_indices, word_positions, word_lengths, is_unique = extract_words(
                samples.cpu().numpy(), lengths.cpu().numpy(), num_threads=4)
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
            return PackedWords(word_feat_matrices, word_lengths, batch_indices, sample_indices, word_positions), is_unique

    def _unpack(self, lm_score: FT, packed_words: PackedWords, batch_size: int) -> FT:
        with torch.no_grad():
            bw = packed_words.word_feat_matrices.size('batch_word')
            p = packed_words.word_feat_matrices.size('position')
            lm_score = lm_score.unflatten('batch', [('batch_word', bw), ('position', p)])
            lm_score = lm_score.sum(dim='position')
            ret = get_zeros(batch_size * self.num_samples)
            bi = packed_words.batch_indices
            si = packed_words.sample_indices
            idx = (bi * self.num_samples + si).rename(None)
            ret.scatter_add_(0, idx, lm_score.rename(None))
            ret = ret.view(batch_size, self.num_samples).refine_names('batch', 'sample')
        return -ret  # NOTE(j_luo) NLL are losses, not scores.

    def _sample(self, label_probs: FT, source_padding: FT) -> Tuple[LT, FT]:
        """Return samples based on `label_probs`."""
        # Ignore padded indices.
        label_probs = label_probs.align_to('batch', 'length', 'label')
        source_padding = source_padding.align_to('batch', 'length')
        # NOTE(j_luo) O is equivalent to None.
        mask = expand_as(source_padding, label_probs)
        source = expand_as(get_tensor([0.0, 0.0, 1.0]).refine_names('label').float(), label_probs)
        # TODO(j_luo) ugly
        label_probs = label_probs.rename(None).masked_scatter(mask.rename(None), source.rename(None))

        # Get packed batches.
        label_distr = Categorical(probs=label_probs.rename(None))
        label_samples = label_distr.sample([self.num_samples]).refine_names('sample', 'batch', 'length')
        label_samples = label_samples.align_to('batch', 'sample', 'length')
        batch_idx = get_named_range(label_samples.size('batch'), 'batch').align_as(label_samples).rename(None)
        length_idx = get_named_range(label_samples.size('length'), 'length').align_as(label_samples).rename(None)
        label_sample_probs = label_probs.rename(None)[batch_idx, length_idx, label_samples.rename(None)]
        label_sample_probs = label_sample_probs.refine_names(*label_samples.names)
        label_sample_log_probs = (1e-8 + label_sample_probs).log()
        label_sample_log_probs = ((~source_padding).align_as(label_sample_log_probs).float()
                                  * label_sample_log_probs).sum(dim='length')
        return label_samples, label_sample_log_probs
