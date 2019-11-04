from devlib import get_zeros
from devlib import get_tensor
from xib.ipa import get_new_style_enum
from arglib import add_argument
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from arglib import g, init_g_attr, not_supported_argument_value
from devlib import freeze
from devlib.named_tensor import gather
from xib.data_loader import DenseIpaBatch, IpaBatch
from xib.ipa import Category, get_enum_by_cat, get_index
from xib.ipa.ipax import CategoryX

from . import FT, LT
from .modules import AdaptLayer, Encoder, Predictor

Cat = Union[Category, CategoryX]


@init_g_attr
class LM(nn.Module):

    add_argument('use_weighted_loss', default=False, dtype=bool, msg='flag to use weighted loss')

    def __init__(self, new_style: 'p', use_weighted_loss: 'p'):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        if use_weighted_loss and not new_style:
            raise ValueError('Must use new_style if using weighted loss')

    def forward(self, batch: IpaBatch) -> Dict[Cat, FT]:
        """
        First encode the `feat_matrix` into a vector `h`, then based on it predict the distributions of features.
        """
        h = self.encoder(batch.feat_matrix, batch.pos_to_predict, batch.source_padding)
        distr = self.predictor(h)
        return distr

    def score(self, batch) -> Dict[Cat, FT]:
        distr = self(batch)
        scores = dict()
        for name, output in distr.items():
            i = get_index(name, new_style=self.new_style)
            target = batch.target_feat[:, i]
            weight = batch.target_weight[:, i]

            if self.new_style:
                e = get_new_style_enum(i)
                mat = get_tensor(e.get_distance_matrix())
                mat = mat[target.rename(None)]
                mat_exp = torch.where(mat > 0, (mat + 1e-8).log(), get_zeros(mat.shape).fill_(-99.9))
                logits = mat_exp + output
                # NOTE(j_luo) For the categories except Ptype, the sums of probs are not 1.0 (they are conditioned on certain values of Ptyle).
                # As a result, we need to incur penalties based on the remaining prob mass as well.
                # Specifically, the remaining prob mass will result in a penalty of 1.0, which is e^(0.0).
                none_probs = (1.0 - output.exp().sum(dim=-1, keepdims=True)).clamp(min=0.0)
                none_penalty = (1e-8 + none_probs).log().align_as(output)
                logits = torch.cat([logits, none_penalty], dim=-1)
                score = torch.logsumexp(logits, dim=-1).exp()
            else:
                log_probs = gather(output, target)
                score = -log_probs
            scores[name] = (score, weight)
        return scores

    @not_supported_argument_value('new_style', True)
    def predict(self, batch, k=-1) -> Dict[Cat, Tuple[FT, LT, np.ndarray]]:
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
class AdaptedLM(LM):

    @not_supported_argument_value('new_style', True)
    def __init__(self, feat_groups, lm_model_path):
        super().__init__()
        saved_dict = torch.load(lm_model_path)
        try:
            self.load_state_dict(saved_dict['model'])
        except RuntimeError as e:
            logging.error(str(e))

        # NOTE(j_luo) We have to map normal feature embedidngs to dense feature embeddings.
        old_weights = saved_dict['model']['encoder.feat_embedding.embed_layer.weight']
        for cat in Category:
            try:
                emb_param = self.encoder.feat_embedding.embed_layer[cat.name]
                e = get_enum_by_cat(cat)
                g_idx = [feat.value.g_idx for feat in e]
                emb_param.data.copy_(old_weights[g_idx])
            except KeyError:
                pass

        freeze(self.encoder)
        freeze(self.predictor)

        self.adapter = AdaptLayer(groups)

    def forward(self, batch: DenseIpaBatch) -> Dict[Category, FT]:
        sfm_adapted = self.adapter(batch.dense_feat_matrix)
        h = self.encoder(sfm_adapted, batch.pos_to_predict, batch.source_padding)
        distr = self.predictor(h)
        return distr
