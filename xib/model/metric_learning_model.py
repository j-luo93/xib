import torch
import torch.nn as nn

from arglib import add_argument, init_g_attr
from xib.data_loader import MetricLearningBatch

from .modules import get_effective_c_idx


@init_g_attr(default='property')
class MetricLearningModel(nn.Module):

    add_argument('num_layers', default=1, dtype=int, msg='number of trainable layers.')

    def __init__(self, hidden_size, emb_groups, num_layers):
        super().__init__()
        effective_num_feat_groups = len(get_effective_c_idx(emb_groups)) + 1  # NOTE(j_luo) +1 due to 'avg' score.
        if num_layers == 1:
            self.regressor = nn.Linear(effective_num_feat_groups, 1)
        else:
            modules = [nn.Linear(effective_num_feat_groups, hidden_size), nn.LeakyReLU(negative_slope=0.1)]
            for _ in range(num_layers - 2):
                modules.append(nn.Linear(hidden_size, hidden_size))
                modules.append(nn.LeakyReLU(negative_slope=0.1))
            modules.append(nn.Linear(hidden_size, 1))
            self.regressor = nn.Sequential(*modules)

    def forward(self, batch: MetricLearningBatch) -> torch.FloatTensor:
        output = self.regressor(batch.normalized_score.rename(None)).view(-1)
        return output
