from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import MultiheadAttention
from torch.distributions.categorical import Categorical

from arglib import add_argument, init_g_attr
from devlib import get_range, get_tensor
from devlib.named_tensor import (adv_index, embed, gather, leaky_relu,
                                 self_attend)
from xib.data_loader import ContinuousTextIpaBatch, IpaBatch
from xib.ipa import (Category, conditions, get_enum_by_cat,
                     no_none_predictions, should_include)

add_argument('num_features', default=10, dtype=int, msg='total number of phonetic features')
add_argument('num_feature_groups', default=10, dtype=int, msg='total number of phonetic feature groups')
add_argument('dim', default=5, dtype=int, msg='dimensionality of feature embeddings')
add_argument('hidden_size', default=5, dtype=int, msg='hidden size')
add_argument('emb_groups', default='pcvdst', dtype=str, msg='what feature groups to embed.')


LT = torch.LongTensor
FT = torch.FloatTensor
BT = torch.BoolTensor


def _get_effective_c_idx(emb_groups):
    if len(set(emb_groups)) != len(emb_groups):
        raise ValueError(f'Duplicate values in emb_groups {emb_groups}.')
    c_idx = list()
    groups = set(emb_groups)
    for cat in Category:
        if cat.name[0].lower() in groups:
            c_idx.append(cat.value)
    return c_idx


@init_g_attr(default='property')
class FeatEmbedding(nn.Module):

    def __init__(self, feat_emb_name, group_name, char_emb_name, num_features, dim, emb_groups, num_feature_groups):
        super().__init__()
        self.embed_layer = nn.Embedding(num_features, dim)
        self.register_buffer('c_idx', get_tensor(_get_effective_c_idx(emb_groups)).refine_names(group_name))
        if len(self.c_idx) > num_feature_groups:
            raise RuntimeError('Something is seriously wrong.')

    @property
    def effective_num_feature_groups(self):
        return len(self.c_idx)

    def forward(self, feat_matrix: LT, padding: BT) -> FT:
        feat_matrix = adv_index(feat_matrix, 'feat_group', self.c_idx)
        feat_emb = embed(self.embed_layer, feat_matrix, self.feat_emb_name)
        feat_emb = feat_emb.flatten([self.group_name, self.feat_emb_name], self.char_emb_name)
        # TODO(j_luo) ugly
        feat_emb = feat_emb.align_to('batch', 'length', self.char_emb_name)
        padding = padding.align_to('batch', 'length')
        feat_emb.rename(None)[padding.rename(None)] = 0.0
        return feat_emb


@init_g_attr(default='property')
class Encoder(nn.Module):

    add_argument('window_size', default=3, dtype=int, msg='window size for the cnn kernel')

    def __init__(self, num_features, dim, window_size, hidden_size, emb_groups):
        super().__init__()

        self.feat_embedding = FeatEmbedding('feat_emb', 'chosen_feat_group', 'char_emb')
        self.cat_dim = dim * self.feat_embedding.effective_num_feature_groups
        # IDEA(j_luo) should I define a Rename layer?
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.cat_dim, self.cat_dim, self.window_size, padding=self.window_size // 2)
        )
        self.linear = nn.Linear(self.cat_dim, self.hidden_size)

    def forward(self, feat_matrix, pos_to_predict, source_padding):
        bs, l, _ = feat_matrix.shape
        feat_emb = self.feat_embedding(feat_matrix, source_padding)
        feat_emb = feat_emb.align_to('batch', 'char_emb', 'length')
        # feat_emb = self.feat_embeddings(feat_matrix).view(bs, l, -1).transpose(1, 2)  # size: bs x D x l
        batch_i = get_range(bs, 1, 0)
        # TODO(j_luo) ugly
        feat_emb.rename(None)[batch_i, :, pos_to_predict.rename(None)] = 0.0
        output = self.conv_layers(feat_emb.rename(None))
        output = output.refine_names('batch', 'char_conv_repr', 'length')  # size: bs x D x l
        output = self.linear(output.align_to(..., 'char_conv_repr'))  # size: bs x l x n_hid
        output = output.refine_names('batch', 'length', 'hidden_repr')
        output = leaky_relu(output, negative_slope=0.1)
        # NOTE(j_luo) This is actually quite wasteful because we are discarding all the irrelevant information, which is computed anyway. This is equivalent to training on ngrams.
        # TODO(j_luo) ugly
        h = output.rename(None)[batch_i, pos_to_predict.rename(None)]
        h = h.refine_names('batch', 'hidden_repr')  # size: bs x n_hid
        return h


@init_g_attr(default='property')
class Predictor(nn.Module):

    def __init__(self, num_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.feat_predictors = nn.ModuleDict()
        for cat in Category:
            e = get_enum_by_cat(cat)
            # NOTE(j_luo) ModuleDict can only hanlde str as keys.
            self.feat_predictors[cat.name] = nn.Linear(hidden_size, len(e))

    def forward(self, h: FT) -> Dict[str, FT]:
        shared_h = leaky_relu(self.linear(h).refine_names(..., 'shared_repr'), negative_slope=0.1)
        ret = dict()
        for name, layer in self.feat_predictors.items():
            cat = getattr(Category, name)
            dim_name = f'{name.lower()}_repr'
            out = layer(shared_h).refine_names(..., dim_name)
            if cat in no_none_predictions:
                index = no_none_predictions[cat]
                out[:, index.f_idx] = -999.9
            ret[cat] = torch.log_softmax(out, dim=-1)
        # Deal with conditions for some categories
        for cat, index in conditions.items():
            # Find out the exact value to be conditioned on.
            condition_cat = Category(index.c_idx)
            condition_log_probs = ret[condition_cat][:, index.f_idx]
            ret[cat] = ret[cat] + condition_log_probs.align_as(ret[cat])

        return ret


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()

    def forward(self, batch: IpaBatch) -> Dict[Category, FT]:
        """
        First encode the `feat_matrix` into a vector `h`, then based on it predict the distributions of features.
        """
        h = self.encoder(batch.feat_matrix, batch.pos_to_predict, batch.source_padding)
        distr = self.predictor(h)
        return distr

    def score(self, batch) -> Dict[Category, FT]:
        distr = self(batch)
        scores = dict()
        for cat, output in distr.items():
            i = cat.value
            target = batch.target_feat[:, i]
            weight = batch.target_weight[:, i]
            log_probs = gather(output, target)
            scores[cat] = (-log_probs, weight)
        return scores

    def predict(self, batch, k=-1) -> Dict[Category, Tuple[FT, LT, np.ndarray]]:
        """
        Predict the top K results for each feature group.
        If k == -1, then everything would be sorted and returned, otherwise take the topk.
        """
        ret = dict()
        distr = self(batch)
        for cat, log_probs in distr.items():
            e = get_enum_by_cat(cat)
            name = cat.name.lower()
            max_k = log_probs.size(name)
            this_k = max_k if k == -1 else min(max_k, k)
            top_values, top_indices = log_probs.topk(this_k, dim=-1)
            top_cats = np.asarray([e.get(i) for i in top_indices.view(-1).cpu().numpy()]).reshape(*top_indices.shape)
            ret[name] = (top_values, top_indices, top_cats)
        return ret


# @dataclass
# class PackedSamples:
#     samples: LT
#     batch_indices: LT
#     sample_indice: LT
#     sample_probs: FT

@dataclass
class PackedWords:
    word_feat_matrices: LT
    word_lengths: LT
    batch_indices: LT
    sample_indices: LT
    word_positions: LT


@init_g_attr(default='property')
class DecipherModel(nn.Module):

    add_argument('adapt_mode', default='none', choices=['none'], dtype=str,
                 msg='how to adapt the features from one language to another')
    add_argument('num_self_attn_layers', default=2, dtype=int, msg='number of self attention layers')
    add_argument('score_per_word', default=1.0, dtype=float, msg='score added for each word')
    add_argument('num_samples', default=100, dtype=int, msg='number of samples per sequence')

    def __init__(self,
                 lm_model: 'a',
                 num_features,
                 dim,
                 emb_groups,
                 adapt_mode,
                 num_self_attn_layers,
                 mode,
                 score_per_word,
                 num_samples):
        super().__init__()
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
        # Get the samples of label sequences first.
        out = self.emb_for_label(batch.feat_matrix, batch.source_padding)
        out = out.align_to('length', 'batch', 'char_emb_for_label')
        for i, layer in enumerate(self.self_attn_layers):
            out, _ = self_attend(layer, out, f'self_attn_repr')
        logits = self.label_predictor(out.rename(None)).refine_names(*out.names)
        logits = logits.rename(**{f'self_attn_repr': 'label'})
        label_probs = logits.log_softmax(dim=-1).exp()
        samples, sample_probs = self._sample(label_probs)

        # Get the lm score.
        packed_words = self._pack(samples, batch.feat_matrix)
        packed_words.word_feat_matrices = self._adapt(packed_words.word_feat_matrices)
        lm_batch = self._prepare_batch(packed_words)  # FIXME(j_luo)  This is actually continous batching.
        scores = self.lm_model.score(lm_batch)
        nlls = list()
        for cat, (nll, _) in scores.items():
            if should_include(self.mode, cat):
                nlls.append(nll)
        nlls = sum(nlls)
        lm_score = self._unpack(nlls, packed_words)  # FIXME(j_luo)

        # Compute word score that corresponds to the number of readable words.
        word_score = self.score_per_word  # FIXME(j_luo)

        bs = batch.feat_matrix.size('batch')
        return {
            'word_score': word_score,
            'lm_score': lm_score
        }

    @staticmethod
    def _prepare_batch(packed_words: PackedWords) -> IpaBatch:

        pass

    @staticmethod
    def _pack(samples: LT, feat_matrix: LT) -> PackedWords:
        # TODO(j_luo) ugly
        with torch.no_grad():
            feat_matrix = feat_matrix.align_to('batch', 'length', 'feat_group')
            samples = samples.align_to('batch', 'sample', 'length')
            batch_indices, sample_indices, word_positions, word_lengths = extract_words(
                samples.cpu().numpy())  # FIXME(j_luo)
            batch_indices = batch_indices.refine_names('batch_word')
            sample_indices = sample_indices.refine_names('batch_word')
            word_positions = word_positions.refine_names('batch_word', 'position')
            word_lengths = word_lengths.refine_name('batch_word')

            key = (
                batch_indices.align_as(word_positions),
                sample_indices.align_as(word_positions),
                word_positions
            )
            word_feat_matrices = feat_matrix.rename(None)[key]
            word_feat_matrices = word_feat_matrices.refine_names('batch_word', 'position', 'feat_group')
            return PackedWords(word_feat_matrices, word_lengths, batch_indices, sample_indices, word_positions)

    @staticmethod
    def _unpack(self, lm_score: FT, packed_words: PackedWords):
        pass

    @staticmethod
    def _sample(label_probs: FT, source_padding: FT) -> Tuple[LT, FT]:
        """Return samples based on `label_probs`."""
        # TODO(j_luo) ugly
        # Ignore padded indices.
        label_probs = label_probs.align_to('batch', 'length', 'label')
        source_padding = source_padding.align_to('batch', 'length')
        label_probs.rename(None)[source_padding.rename(None).unsqueeze(dim=-1)] = [0.0, 0.0, 1.0] # NOTE(j_luo) O is equivalent to None.

        # Get packed batches.
        label_distr = Categorical(probs=label_probs.rename(None))
        samples = label_distr.sample([self.num_samples]).refine_names('sample', 'batch', 'length')
        samples = samples.align_to('batch', 'sample', 'length')
        batch_idx = get_range(samples.size('batch'), 3, 1)
        sample_idx = get_range(samples.size('sample'), 3, 0)
        sample_probs = label_probs.rename(None)[sample_idx, batch_idx, samples.rename(None)]
        sample_probs = sample_probs.refine_names('sample', 'batch', 'length')
        sample_probs = (source_padding.align_as(sample_probs).float() * sample_probs).sum(dim='length')
        return samples, sample_probs
        # samples = samples.cpu().numpy()
        # unique_samples, batch_indices, sample_indices = batch_unique(samples, 2)  # FIXME(j_luo)
        # unique_samples = unique_samples.refine_names('packed_batch_x_sample', 'length')
        # batch_indices = batch_indices.refine_names('packed_batch_x_sample')
        # sample_indices = sample_indices.refine_names('packed_batch_x_sample')

        # # Get packed probs.
        # sample_probs = samples.rename(None)[batch_indices, sample_indices]
        # sample_probs = sample_probs.refine_names('packed_batch_x_sample', 'length')
        # source_padding = source_padding.rename(None)[batch_indices]
        # # IDEA(j_luo) Can I bind name to a fixed number?
        # source_padding = source_padding.refine_names('packed_batch_x_sample', 'length')
        # sample_probs = (sample_probs * source_padding.float()).sum(dim='length')
        # packed_samples = PackedSamples(unique_samples, batch_indices, sample_indice, sample_probs)
        # return packed_samples

        # sample_probs = label_probs.refine(None).gather(2, samples.rename(None))
        # sample_probs = sample_probs.refine_names('batch', 'length', 'sample')
        # return samples, sample_probs
