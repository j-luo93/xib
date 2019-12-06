import logging
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from dev_misc.arglib import add_argument, g, not_supported_argument_value
from dev_misc.devlib import BaseBatch, batch_class, get_tensor, get_zeros
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import freeze
from xib.data_loader import DenseIpaBatch, IpaBatch
from xib.ipa import Category, get_enum_by_cat, get_index, get_new_style_enum
from xib.ipa.ipax import CategoryX

from . import FT, LT
from .modules import AdaptLayer, CbowEncoder, Encoder, Predictor

Cat = Union[Category, CategoryX]


class LM(nn.Module):

    add_argument('weighted_loss', default='', dtype=str,
                 choices=['', 'mr', 'ot'], msg='what type of weighted loss to use')
    add_argument('use_cbow_encoder', dtype=bool, default=True, msg='Flag to use cbow encoder.')

    def _get_encoder(self, dropout: float = None, hidden_size: int = None, dim: int = None):
        dropout = dropout or g.dropout
        hidden_size = hidden_size or g.hidden_size
        dim = dim or g.dim
        encoder_cls = CbowEncoder if g.use_cbow_encoder else Encoder
        return encoder_cls(dropout=dropout, hidden_size=hidden_size, dim=dim)

    def __init__(self):
        super().__init__()
        self.encoder = self._get_encoder()
        self.predictor = Predictor()
        # if weighted_loss and not new_style:
        #     raise ValueError('Must use new_style if using weighted loss')

    def forward(self, batch: IpaBatch) -> Dict[Cat, FT]:
        """
        First encode the `feat_matrix` into a vector `h`, then based on it predict the distributions of features.
        """
        h = self.encoder(batch.feat_matrix, batch.pos_to_predict, batch.source_padding)
        distr = self.predictor(h)
        return distr

    def score(self, batch) -> Dict[Cat, FT]:
        distr = self(batch)
        return self.score_distr(distr, batch)

    def score_distr(self, distr: Dict[Cat, FT], batch: IpaBatch) -> Dict[Cat, FT]:
        scores = dict()
        for name, output in distr.items():
            i = get_index(name, new_style=g.new_style)
            target = batch.target_feat[:, i]
            weight = batch.target_weight[:, i]

            if g.weighted_loss == '':
                # log_probs = gather(output, target)
                log_probs = output.gather(name.value, target)
                score = -log_probs
            else:
                e = get_new_style_enum(i)
                mat = get_tensor(e.get_distance_matrix())
                mat = mat[target.rename(None)]
                if g.weighted_loss == 'mr':
                    mat_exp = torch.where(mat > 0, (mat + 1e-8).log(), get_zeros(mat.shape).fill_(-99.9))
                    logits = mat_exp + output
                    # NOTE(j_luo) For the categories except Ptype, the sums of probs are not 1.0 (they are conditioned on certain values of Ptyle).
                    # As a result, we need to incur penalties based on the remaining prob mass as well.
                    # Specifically, the remaining prob mass will result in a penalty of 1.0, which is e^(0.0).
                    none_probs = (1.0 - output.exp().sum(dim=-1, keepdims=True)).clamp(min=0.0)
                    none_penalty = (1e-8 + none_probs).log().align_as(output)
                    logits = torch.cat([logits, none_penalty], dim=-1)
                    score = torch.logsumexp(logits, dim=-1).exp()
                elif g.weighted_loss == 'ot':
                    if not self.training:
                        raise RuntimeError('Cannot use OT for training.')

                    probs = output.exp()
                    # We have to incur penalties based on the remaining prob mass as well.
                    none_probs = (1.0 - probs.sum(dim=-1, keepdims=True)).clamp(min=0.0)
                    mat = torch.cat([mat, get_tensor(torch.ones_like(none_probs.rename(None)))], dim=-1)
                    probs = torch.cat([probs, none_probs], dim=-1)
                    score = (mat * probs).sum(dim=-1)
                else:
                    raise ValueError(f'Cannot recognize {self.weighted_loss}.')
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


@batch_class
class AdaptLMReturn(BaseBatch):
    distr: Dict[Category, FT]
    gate_logits: Optional[FT] = None
    distr_noise: Dict[Category, FT] = None


class AdaptLM(LM):

    add_argument('use_prior', dtype=bool, default=False, msg='Flag to use prior.')
    add_argument('prior_value', dtype=float, default=0.5, msg='Value for prior.')
    add_argument('use_moe', dtype=bool, default=False, msg='Flag to use MoE.')

    @not_supported_argument_value('new_style', True)
    def __init__(self):
        super().__init__()
        saved_dict = torch.load(g.lm_model_path)
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

        self.adapter = AdaptLayer()

        if g.use_prior or g.use_moe:
            noise_hs = 10
            noise_dim = 10
            self.noise_encoder = self._get_encoder(hidden_size=noise_hs, dim=noise_dim)
            self.noise_predictor = Predictor(hidden_size=noise_hs)
            if g.use_moe:
                self.moe_gate = nn.Linear(noise_hs + g.hidden_size, 2)

    def forward(self, batch: DenseIpaBatch) -> AdaptLMReturn:
        sfm_adapted = self.adapter(batch.dense_feat_matrix)
        h = self.encoder(sfm_adapted, batch.pos_to_predict, batch.source_padding)
        distr = self.predictor(h)

        if g.use_prior:
            if g.use_moe:
                h_noise = self.noise_encoder(batch.dense_feat_matrix, batch.pos_to_predict, batch.source_padding)
                distr_noise = self.noise_predictor(h_noise)

                gate_logits = self.moe_gate(torch.cat([h, h_noise], dim=-1))  # / self._temp
                gate_log_probs = gate_logits.log_softmax(dim=-1)

                return AdaptLMReturn(distr, gate_logits, distr_noise)
            else:
                h_noise = self.noise_encoder(batch.dense_feat_matrix, batch.pos_to_predict, batch.source_padding)
                distr_noise = self.noise_predictor(h_noise)
                for cat in distr:
                    d = distr[cat]
                    d_noise = distr_noise[cat]
                    stacked = torch.stack([d + math.log(g.prior_value),
                                           d_noise + math.log(1.0 - g.prior_value)],
                                          new_name='prior')
                    new_d = stacked.logsumexp(dim='prior')
                    distr[cat] = new_d

        return AdaptLMReturn(distr)

    def score(self, batch) -> AdaptLMReturn:
        ret = self(batch)
        return AdaptLMReturn(self.score_distr(ret.distr, batch), ret.gate_logits, self.score_distr(ret.distr_noise, batch))
