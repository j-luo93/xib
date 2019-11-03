from typing import Dict

import inflection
import torch
import torch.nn as nn

from arglib import add_argument, init_g_attr, not_supported_argument_value
from devlib import check_explicit_arg, get_range, get_tensor
from devlib.named_tensor import adv_index, embed, leaky_relu
from xib.ipa import (Category, Name, conditions, get_enum_by_cat,
                     get_needed_categories, get_none_index,
                     no_none_predictions, should_include, should_predict_none)

from . import BT, FT, LT


# TODO(j_luo) should not have init_g_attr here.
@init_g_attr(default='property')
class FeatEmbedding(nn.Module):

    def __init__(self, feat_emb_name, group_name, char_emb_name, num_features, dim, groups, num_feature_groups, new_style):
        super().__init__()
        # self.register_buffer('c_idx', get_tensor(get_effective_c_idx(groups)).refine_names(group_name))
        # if len(self.c_idx) > num_feature_groups:
        #     raise RuntimeError('Something is seriously wrong.')
        self.embed_layer = self._get_embeddings()
        cat_enum_pairs = get_needed_categories(groups, new_style=new_style, breakdown=True)
        if new_style:
            self.effective_num_feature_groups = sum([e.num_groups() for _, e in cat_enum_pairs])
        else:
            self.effective_num_feature_groups = len(cat_enum_pairs)

    def _get_embeddings(self):
        return nn.Embedding(self.num_features, self.dim)

    def forward(self, feat_matrix: LT, padding: BT) -> FT:
        # feat_matrix = adv_index(feat_matrix, 'feat_group', self.c_idx)
        feat_emb = embed(self.embed_layer, feat_matrix, self.feat_emb_name)
        feat_emb = feat_emb.flatten([self.group_name, self.feat_emb_name], self.char_emb_name)
        # TODO(j_luo) ugly
        feat_emb = feat_emb.align_to('batch', 'length', self.char_emb_name)
        padding = padding.align_to('batch', 'length')
        feat_emb.rename(None)[padding.rename(None)] = 0.0
        return feat_emb


@init_g_attr(default='property')
class DenseFeatEmbedding(FeatEmbedding):

    @not_supported_argument_value('new_style', True)
    def _get_embeddings(self):
        emb_dict = dict()
        for cat in Category:
            if should_include(self.groups, cat):
                e = get_enum_by_cat(cat)
                nf = len(e)
                emb_dict[cat.name] = nn.Parameter(torch.zeros(nf, self.dim))
        return nn.ParameterDict(emb_dict)

    def forward(self, dense_feat_matrices: Dict[Category, FT], padding: BT) -> FT:
        embs = list()
        for cat in Category:
            if cat.name in self.embed_layer and cat in dense_feat_matrices:
                sfm = dense_feat_matrices[cat]
                emb_param = self.embed_layer[cat.name]
                sfm = sfm.align_to('batch', 'length', ...)
                emb = sfm @ emb_param
                embs.append(emb)
        feat_emb = torch.cat(embs, dim=-1).refine_names('batch', 'length', self.char_emb_name)
        return feat_emb


@init_g_attr(default='property')
class Encoder(nn.Module):

    add_argument('window_size', default=3, dtype=int, msg='window size for the cnn kernel')
    add_argument('dense_input', default=False, dtype=bool, msg='flag to dense input feature matrices')

    def __init__(self, num_features, dim, window_size, hidden_size, groups, dense_input):
        super().__init__()

        emb_cls = DenseFeatEmbedding if dense_input else FeatEmbedding
        self.feat_embedding = emb_cls('feat_emb', 'chosen_feat_group', 'char_emb')
        self.cat_dim = dim * self.feat_embedding.effective_num_feature_groups
        # IDEA(j_luo) should I define a Rename layer?
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.cat_dim, self.cat_dim, self.window_size, padding=self.window_size // 2)
        )
        self.linear = nn.Linear(self.cat_dim, self.hidden_size)

    def forward(self, feat_matrix, pos_to_predict, source_padding):
        bs = source_padding.size('batch')
        l = source_padding.size('length')
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

    def __init__(self, hidden_size, groups, new_style):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.feat_predictors = nn.ModuleDict()
        for e in get_needed_categories(groups, new_style=new_style, breakdown=new_style):
            # NOTE(j_luo) ModuleDict can only handle str as keys.
            self.feat_predictors[e.__name__] = nn.Linear(hidden_size, len(e))
        # If new_style, we need to get the necessary indices to convert the breakdown groups into the original feature groups.
        if new_style:
            self.conversion_idx = dict()
            for e in get_needed_categories(self.groups, new_style=True, breakdown=False):
                if e.num_groups() > 1:
                    cat_idx = list()
                    for feat in e:
                        feat_cat_idx = list()
                        feat = feat.value
                        for basic_feat in feat:
                            idx = basic_feat.value
                            feat_cat_idx.append(idx)
                        cat_idx.append(feat_cat_idx)
                    cat_idx = get_tensor(cat_idx).refine_names('new_style_idx', 'old_style_idx')
                    self.conversion_idx[e.__name__] = cat_idx

    def forward(self, h: FT) -> Dict[str, FT]:
        shared_h = leaky_relu(self.linear(h).refine_names(..., 'shared_repr'), negative_slope=0.1)
        ret = dict()
        for name, layer in self.feat_predictors.items():
            dim_name = f'{inflection.underscore(name)}_repr'
            out = layer(shared_h).refine_names(..., dim_name)
            if not should_predict_none(name, new_style=self.new_style):
                f_idx = get_none_index(name)
                out[:, f_idx] = -999.9
            ret[Name(name, 'camel')] = out

        # Compose probs for complex feature groups if possible.
        if self.new_style:
            for e in get_needed_categories(self.groups, new_style=True, breakdown=False):
                if e.num_groups() > 1:
                    assert e not in ret
                    part_tensors = [ret[part_enum.get_name()] for part_enum in e.parts()]
                    parts = list()
                    for i, part_tensor in enumerate(part_tensors):
                        conversion = self.conversion_idx[e.get_name().value][:, i]
                        bs = len(part_tensor)
                        part = part_tensor.rename(None).gather(1, conversion.rename(None).expand(bs, -1))
                        parts.append(part)
                    parts = torch.stack(parts, dim=-1)
                    ret[e.get_name()] = parts.sum(dim=-1)
                    for part_cat in e.parts():
                        del ret[part_cat.get_name()]
        for name in ret:
            ret[name] = torch.log_softmax(ret[name], dim=-1)

        # Deal with conditions for some categories
        for cat, index in conditions.items():
            # Find out the exact value to be conditioned on.
            # TODO(j_luo) ugly Category call.
            condition_e = get_enum_by_cat(Category(index.c_idx))
            condition_name = condition_e.__name__ + ('X' if self.new_style else '')
            cat_name = get_enum_by_cat(cat).__name__ + ('X' if self.new_style else '')

            condition_name = Name(condition_name, 'camel')
            cat_name = Name(cat_name, 'camel')
            condition_log_probs = ret[condition_name][:, index.f_idx]
            # condition_log_probs.align_as(ret[cat_name])
            ret[cat_name] = ret[cat_name] + condition_log_probs.unsqueeze(dim=1)
        return ret


class AdaptLayer(nn.Module):

    @not_supported_argument_value('new_style', True)
    def __init__(self, groups: str):
        super().__init__()
        param_dict = dict()
        for cat in Category:
            if should_include(groups, cat):
                e = get_enum_by_cat(cat)
                nf = len(e)
                param = nn.Parameter(torch.zeros(nf, nf))
                param_dict[cat.name] = param
        self.adapters = nn.ParameterDict(param_dict)

    def alignment(self, cat_name: str) -> FT:
        param = self.adapters[cat_name]
        alignment = param.log_softmax(dim=0).exp()
        return alignment

    def forward(self, dense_feat_matrices: Dict[Category, FT]) -> Dict[Category, FT]:
        ret = dict()
        for cat, sfm in dense_feat_matrices.items():
            if cat.name in self.adapters:
                alignment = self.alignment(cat.name)
                sfm_adapted = sfm @ alignment
                ret[cat] = sfm_adapted.refine_names('batch', 'length', f'{cat.name}_feat_adapted')
        return ret
