from typing import Dict

import torch
import torch.nn as nn

from arglib import add_argument, init_g_attr
from devlib import get_range, get_tensor
from devlib.named_tensor import adv_index, embed, leaky_relu
from xib.ipa import Category, conditions, get_enum_by_cat, no_none_predictions

from . import BT, FT, LT


def get_effective_c_idx(emb_groups):
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
        self.register_buffer('c_idx', get_tensor(get_effective_c_idx(emb_groups)).refine_names(group_name))
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
