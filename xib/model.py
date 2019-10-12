from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn as nn
from arglib import add_argument, init_g_attr
from devlib import get_range
from devlib.named_tensor import NamedTensor
from xib.ipa import Category, conditions, get_enum_by_cat, no_none_predictions


@init_g_attr(default='property')
class Encoder(nn.Module):

    def __init__(self, num_features, num_feature_groups, dim, window_size, hidden_size):
        super().__init__()
        self.cat_dim = dim * num_feature_groups
        self.feat_embeddings = nn.Embedding(self.num_features, self.dim)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.cat_dim, self.cat_dim, self.window_size, padding=self.window_size // 2)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.cat_dim, self.hidden_size),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, feat_matrix, pos_to_predict):
        bs, l, _ = feat_matrix.shape
        feat_emb = self.feat_embeddings(feat_matrix).view(bs, l, -1).transpose(1, 2)  # size: bs x D x l
        batch_i = get_range(bs, 1, 0)
        feat_emb[batch_i, :, pos_to_predict] = 0.0
        output = self.conv_layers(feat_emb)  # size: bs x D x l
        output = self.linear_layers(output.transpose(1, 2))  # size: bs x l x n_hid
        # NOTE(j_luo) This is actually quite wasteful because we are discarding all the irrelevant information, which is computed anyway. This is equivalent to training on ngrams.
        h = output[batch_i, pos_to_predict]  # size: bs x n_hid
        return h


@init_g_attr(default='property')
class Predictor(nn.Module):

    def __init__(self, num_features, hidden_size, window_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
        )
        self.feat_predictors = nn.ModuleDict()
        for cat in Category:
            e = get_enum_by_cat(cat)
            # NOTE(j_luo) ModuleDict can only hanlde str as keys.
            self.feat_predictors[cat.name] = nn.Linear(hidden_size, len(e))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_h = self.layers(h)
        ret = dict()
        for name, layer in self.feat_predictors.items():
            cat = getattr(Category, name)
            out = layer(shared_h)
            if cat in no_none_predictions:
                index = no_none_predictions[cat]
                out[:, index.f_idx] = -999.9
            ret[cat] = torch.log_softmax(out, dim=-1)
        # Deal with conditions for some categories
        for cat, index in conditions.items():
            # Find out the exact value to be conditioned on.
            condition_log_probs = ret[cat][:, index.f_idx]
            ret[cat] = ret[cat] + condition_log_probs.unsqueeze(dim=-1)

        return ret


@init_g_attr(default='property')
class Model(nn.Module):

    add_argument('num_features', default=10, dtype=int, msg='total number of phonetic features')
    add_argument('num_feature_groups', default=10, dtype=int, msg='total number of phonetic feature groups')
    add_argument('dim', default=5, dtype=int, msg='dimensionality of feature embeddings and number of hidden units')
    add_argument('hidden_size', default=5, dtype=int, msg='hidden size')

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
            log_probs = output.gather(1, target.view(-1, 1)).view(-1)
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
            log_probs = NamedTensor(log_probs, names=['batch', name])
            max_k = log_probs.size(name)
            this_k = max_k if k == -1 else min(max_k, k)
            top_values, top_indices = log_probs.topk(this_k, dim=-1)
            top_cats = np.asarray([e.get(i) for i in top_indices.view(-1).cpu().numpy()]).reshape(*top_indices.shape)
            ret[name] = (top_values, top_indices, top_cats)
        return ret
