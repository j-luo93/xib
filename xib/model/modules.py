from typing import Optional
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from dev_misc import g
from dev_misc.arglib import add_argument, not_supported_argument_value
from dev_misc.devlib import get_range, get_tensor
from dev_misc.devlib.named_tensor import adv_index, embed
from dev_misc.utils import check_explicit_arg
from xib.ipa import (Category, Name, conditions, get_enum_by_cat,
                     get_needed_categories, get_new_style_enum, get_none_index,
                     no_none_predictions, should_include, should_predict_none)
from xib.ipa.ipax import conversions

from . import BT, FT, LT


def get_effective_c_idx():
    if len(set(g.feat_groups)) != len(g.feat_groups):
        raise ValueError(f'Duplicate values in feat_groups {g.feat_groups}.')
    c_idx = list()
    feat_groups = set(g.feat_groups)
    for cat in Category:
        if cat.name[0].lower() in feat_groups:
            c_idx.append(cat.value)
    return c_idx


class FeatEmbedding(nn.Module):

    def __init__(self, feat_emb_name, group_name, char_emb_name):
        super().__init__()
        self.feat_emb_name = feat_emb_name
        self.group_name = group_name
        self.char_emb_name = char_emb_name

        self.embed_layer = self._get_embeddings()
        self.register_buffer('c_idx', get_tensor(get_effective_c_idx()).refine_names('chosen_feat_group'))
        cat_enum_pairs = get_needed_categories(g.feat_groups, new_style=g.new_style, breakdown=g.new_style)
        if g.new_style:
            self.effective_num_feature_groups = sum([e.num_groups() for e in cat_enum_pairs])
            simple_conversions = np.zeros([g.num_features], dtype='int64')
            max_len = max(len(new_feat.value) for new_feat in conversions.values() if new_feat.value.is_complex())
            complex_conversions = np.zeros([g.num_features, max_len], dtype='int64')
            for old_feat, new_feat in conversions.items():
                if new_feat.value.is_complex():
                    l = len(new_feat.value)
                    complex_conversions[old_feat.value.g_idx, :l] = [x.value.g_idx for x in new_feat.value]
                else:
                    simple_conversions[old_feat.value.g_idx] = new_feat.value.g_idx
            self.simple_conversions = get_tensor(simple_conversions)
            self.complex_conversions = get_tensor(complex_conversions)
        else:
            self.effective_num_feature_groups = len(cat_enum_pairs)

    def _get_embeddings(self):
        return nn.Embedding(g.num_features, g.dim)

    def forward(self, feat_matrix: LT, padding: BT, masked_positions: Optional[LT] = None) -> FT:
        feat_matrix = adv_index(feat_matrix, 'feat_group', self.c_idx)
        # Convert old style to new style ipa features.
        if g.new_style:
            new_feat_matrix = list()
            for c_idx, one_feat_group in zip(self.c_idx.unbind(dim=self.group_name), feat_matrix.unbind(dim=self.group_name)):
                one_feat_group = one_feat_group.rename(None)
                new_enum = get_new_style_enum(c_idx.item())
                l = new_enum.num_groups()
                if l > 1:
                    new_feat_matrix.append(self.complex_conversions[one_feat_group][..., :l])
                else:
                    new_feat_matrix.append(self.simple_conversions[one_feat_group].unsqueeze(dim=-1))
            new_feat_matrix = torch.cat(new_feat_matrix, dim=-1).refine_names(*feat_matrix.names)
            feat_matrix = new_feat_matrix
        feat_emb = embed(self.embed_layer, feat_matrix, self.feat_emb_name)
        feat_emb = feat_emb.flatten([self.group_name, self.feat_emb_name], self.char_emb_name)
        feat_emb = feat_emb.align_to('batch', 'length', self.char_emb_name)
        padding = padding.align_to('batch', 'length')
        feat_emb.rename(None)[padding.rename(None)] = 0.0

        if masked_positions is not None:
            batch_i = get_range(padding.size('batch'), 1, 0)
            feat_emb = feat_emb.align_to('batch', 'char_emb', 'length')
            # feat_emb = self.feat_embeddings(feat_matrix).view(bs, l, -1).transpose(1, 2)  # size: bs x D x l
            # TODO(j_luo) ugly
            feat_emb.rename(None)[batch_i, :, masked_positions.rename(None)] = 0.0
        return feat_emb


class DenseFeatEmbedding(FeatEmbedding):

    @not_supported_argument_value('new_style', True)
    def _get_embeddings(self):
        emb_dict = dict()
        for cat in Category:
            if should_include(g.feat_groups, cat):
                e = get_enum_by_cat(cat)
                nf = len(e)
                emb_dict[cat.name] = nn.Parameter(torch.zeros(nf, g.dim))
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


class Encoder(nn.Module):

    add_argument('window_size', default=3, dtype=int, msg='window size for the cnn kernel')
    add_argument('dense_input', default=False, dtype=bool, msg='flag to dense input feature matrices')

    def __init__(self):
        super().__init__()

        emb_cls = DenseFeatEmbedding if g.dense_input else FeatEmbedding
        self.feat_embedding = emb_cls('feat_emb', 'chosen_feat_group', 'char_emb')
        self.cat_dim = g.dim * self.feat_embedding.effective_num_feature_groups
        # IDEA(j_luo) should I define a Rename layer?
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.cat_dim, self.cat_dim, g.window_size, padding=g.window_size // 2)
        )
        self.linear = nn.Linear(self.cat_dim, g.hidden_size)

    def forward(self, feat_matrix: LT, pos_to_predict: LT, source_padding: BT) -> FT:
        bs = source_padding.size('batch')
        l = source_padding.size('length')
        batch_i = get_range(bs, 1, 0)
        feat_emb = self.feat_embedding(feat_matrix, source_padding, masked_positions=pos_to_predict)
        output = self.conv_layers(feat_emb.rename(None))
        output = output.refine_names('batch', 'char_conv_repr', 'length')  # size: bs x D x l
        output = self.linear(output.align_to(..., 'char_conv_repr'))  # size: bs x l x n_hid
        output = output.refine_names('batch', 'length', 'hidden_repr')
        output = nn.functional.leaky_relu(output, negative_slope=0.1)
        # NOTE(j_luo) This is actually quite wasteful because we are discarding all the irrelevant information, which is computed anyway. This is equivalent to training on ngrams.
        # TODO(j_luo) ugly
        h = output.rename(None)[batch_i, pos_to_predict.rename(None)]
        h = h.refine_names('batch', 'hidden_repr')  # size: bs x n_hid
        return h


class Predictor(nn.Module):

    def __init__(self, hidden_size: Optional[int] = None):
        hidden_size = hidden_size or g.hidden_size
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.feat_predictors = nn.ModuleDict()
        for e in get_needed_categories(g.feat_groups, new_style=g.new_style, breakdown=g.new_style):
            # NOTE(j_luo) ModuleDict can only handle str as keys.
            self.feat_predictors[e.__name__] = nn.Linear(hidden_size, len(e))
        # If new_style, we need to get the necessary indices to convert the breakdown groups into the original feature groups.
        if g.new_style:
            self.conversion_idx = dict()
            for e in get_needed_categories(g.feat_groups, new_style=True, breakdown=False):
                if e.num_groups() > 1:
                    cat_idx = list()
                    for feat in e:
                        feat_cat_idx = list()
                        feat = feat.value
                        for basic_feat in feat:
                            auto_index = basic_feat.value
                            feat_cat_idx.append(auto_index.f_idx)
                        cat_idx.append(feat_cat_idx)
                    cat_idx = get_tensor(cat_idx).refine_names('new_style_idx', 'old_style_idx')
                    self.conversion_idx[e.__name__] = cat_idx

    def forward(self, h: FT) -> Dict[str, FT]:
        shared_h = nn.functional.leaky_relu(self.linear(h).refine_names(..., 'shared_repr'), negative_slope=0.1)
        ret = dict()
        for name, layer in self.feat_predictors.items():
            out = layer(shared_h).refine_names(..., name)
            if not should_predict_none(name, new_style=g.new_style):
                f_idx = get_none_index(name)
                out[:, f_idx] = -999.9
            ret[Name(name, 'camel')] = out

        # Compose probs for complex feature groups if possible.
        if g.new_style:
            for e in get_needed_categories(g.feat_groups, new_style=True, breakdown=False):
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
                    dim_name = e.get_name().value
                    ret[e.get_name()] = parts.sum(dim=-1).refine_names('batch', dim_name)
                    for part_cat in e.parts():
                        del ret[part_cat.get_name()]
        for name in ret:
            ret[name] = torch.log_softmax(ret[name], dim=-1)

        # Deal with conditions for some categories
        for cat, index in conditions.items():
            if should_include(g.feat_groups, cat):
                # Find out the exact value to be conditioned on.
                # TODO(j_luo) ugly Category call.
                condition_e = get_enum_by_cat(Category(index.c_idx))
                condition_name = condition_e.__name__ + ('X' if g.new_style else '')
                cat_name = get_enum_by_cat(cat).__name__ + ('X' if g.new_style else '')

                condition_name = Name(condition_name, 'camel')
                cat_name = Name(cat_name, 'camel')
                condition_log_probs = ret[condition_name][..., index.f_idx]
                # condition_log_probs.align_as(ret[cat_name])
                ret[cat_name] = ret[cat_name] + condition_log_probs.rename(None).unsqueeze(dim=-1)
        return ret


class AdaptLayer(nn.Module):

    @not_supported_argument_value('new_style', True)
    def __init__(self):
        super().__init__()
        param_dict = dict()
        for cat in Category:
            if should_include(g.feat_groups, cat):
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
