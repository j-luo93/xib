from devlib.named_tensor import adv_index
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import MultiheadAttention

from arglib import add_argument, init_g_attr
from devlib import get_range, get_tensor
from devlib.named_tensor import embed, self_attend, leaky_relu, gather
from xib.data_loader import Batch
from xib.ipa import (Category, conditions, get_enum_by_cat,
                     no_none_predictions, should_include)

add_argument('num_features', default=10, dtype=int, msg='total number of phonetic features')
add_argument('num_feature_groups', default=10, dtype=int, msg='total number of phonetic feature groups')
add_argument('dim', default=5, dtype=int, msg='dimensionality of feature embeddings')
add_argument('hidden_size', default=5, dtype=int, msg='hidden size')
add_argument('emb_groups', default='pcvdst', dtype=str, msg='what feature groups to embed.')


Tensor = torch.Tensor


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
class Encoder(nn.Module):

    def __init__(self, num_features, num_feature_groups, dim, window_size, hidden_size, emb_groups):
        super().__init__()
        self.register_buffer('c_idx', get_tensor(_get_effective_c_idx(emb_groups)).refine_names('chosen_feat_group'))
        if len(self.c_idx) > num_feature_groups:
            raise RuntimeError('Something is seriously wrong.')

        self.cat_dim = dim * len(self.c_idx)
        self.feat_embeddings = nn.Embedding(self.num_features, self.dim)
        # IDEA(j_luo) should I define a Rename layer?
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.cat_dim, self.cat_dim, self.window_size, padding=self.window_size // 2)
        )
        self.linear = nn.Linear(self.cat_dim, self.hidden_size)

    def forward(self, feat_matrix, pos_to_predict):
        bs, l, _ = feat_matrix.shape
        feat_matrix = adv_index(feat_matrix, 'feat_group', self.c_idx)
        feat_emb = embed(self.feat_embeddings, feat_matrix, 'feat_emb')
        feat_emb = feat_emb.flatten(['chosen_feat_group', 'feat_emb'], 'char_emb')
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

    def __init__(self, num_features, hidden_size, window_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.feat_predictors = nn.ModuleDict()
        for cat in Category:
            e = get_enum_by_cat(cat)
            # NOTE(j_luo) ModuleDict can only hanlde str as keys.
            self.feat_predictors[cat.name] = nn.Linear(hidden_size, len(e))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
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


@init_g_attr(default='property')
class Model(nn.Module):

    def __init__(self, num_features, num_feature_groups, dim, window_size):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()

    def forward(self, batch) -> Dict[Category, torch.Tensor]:
        """
        First encode the `feat_matrix` into a vector `h`, then based on it predict the distributions of features.
        """
        h = self.encoder(batch.feat_matrix, batch.pos_to_predict)
        distr = self.predictor(h)
        return distr

    def score(self, batch) -> Dict[Category, torch.Tensor]:
        distr = self(batch)
        scores = dict()
        for cat, output in distr.items():
            i = cat.value
            target = batch.target_feat[:, i]
            weight = batch.target_weight[:, i]
            log_probs = gather(output, target)
            scores[cat] = (-log_probs, weight)
        return scores

    def predict(self, batch, k=-1) -> Dict[Category, Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
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


@init_g_attr(default='property')
class DecipherModel(nn.Module):

    add_argument('adapt_mode', default='none', choices=['none'], dtype=str,
                 msg='how to adapt the features from one language to another')
    add_argument('num_self_attn_layers', default=2, dtype=int, msg='number of self attention layers')
    add_argument('score_per_word', default=1.0, dtype=float, msg='score added for each word')

    def __init__(self, lm_model: 'a', num_features, dim, emb_groups, adapt_mode, num_self_attn_layers, mode, score_per_word):
        super().__init__()
        # NOTE(j_luo) I'm keeping two embeddings for now, now for LM evaluation and the other for prediction BIOs.
        self.emb_for_label = nn.Embedding(num_features, dim)
        self.emb_for_lm = nn.Embedding(num_features, dim)

        self.self_attn_layers = nn.ModuleList()
        for _ in range(num_self_attn_layers):
            self.self_attn_layers.append(MultiheadAttention(dim, 8))

        cat_dim = dim * len(_get_effective_c_idx(emb_groups))
        self.label_predictor = nn.Sequential(
            nn.Linear(cat_dim, cat_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(cat_dim, 3)  # BIO.
        )

    def _adapt(self, packed_feat_matrix: Tensor) -> Tensor:
        if self.adapt_mode == 'none':
            return packed_feat_matrix
        else:
            raise NotImplementedError()

    def forward(self, batch: Batch):
        # Get the samples of label sequences first.
        out = embed(self.emb_for_label, batch.feat_matrix, 'feat_dim_for_label')
        out = out.flatten(['feat_group', 'feat_dim_for_label'], 'char_dim_for_label')
        out = out.align_to(['length', 'batch', 'char_dim_for_label'])
        for layer in self.self_attn_layers:
            out, _ = self_attend(layer, out)
        label_probs = out.log_softmax(dim=-1).exp()
        label_seq_samples, label_seq_sample_probs = DecipherModel._sample(label_probs)  # FIXME(j_luo)

        # Get the lm score.
        packed_feat_matrix, orig_idx = self._pack(label_seq_samples, batch.feat_matrix)  # FIXME(j_luo)
        packed_feat_matrix = self._adapt(packed_feat_matrix)
        lm_batch = self._prepare_batch(packed_feat_matrix)  # FIXME(j_luo)
        scores = self.lm_model.score(lm_batch)  # FIXME(j_luo) make sure that is named.
        nlls = list()
        for cat, (nll, _) in scores.items():
            if should_include(self.mode, cat):
                nlls.append(nll)
        nlls = sum(nlls)
        lm_score = self._unpack(nlls, orig_idx)  # FIXME(j_luo)

        # Compute word score that corresponds to the number of readable words.
        word_score = self.score_per_word  # FIXME(j_luo)

        bs = batch.feat_matrix.size('batch')
        return {
            'word_score': word_score,
            'lm_score': lm_score
        }

    @staticmethod
    def _prepare_batch(packed_feat_matrix: Tensor) -> Batch:
        pass

    @staticmethod
    def _pack(label_seq_samples: Tensor, feat_matrix: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @staticmethod
    def _unpack(self, lm_score: Tensor, orig_idx: Tensor) -> Tensor:
        pass

    @staticmethod
    def _sample(label_probs: Tensor) -> Tuple[Tensor, Tensor]:
        breakpoint() # DEBUG(j_luo)
        pass
