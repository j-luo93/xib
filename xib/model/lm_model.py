from typing import Dict, Tuple

import numpy as np
import torch.nn as nn

from devlib.named_tensor import gather
from xib.data_loader import IpaBatch
from xib.ipa import Category, get_enum_by_cat

from . import FT, LT
from .modules import Encoder, Predictor


class LMModel(nn.Module):

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
