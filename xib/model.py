import torch
import torch.nn as nn

from arglib import add_argument, init_g_attr
from devlib import get_range


class Encoder(nn.Module):

    def __init__(self, num_features, num_feature_groups, dim, window_size):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.num_feature_groups = num_feature_groups
        self.window_size = window_size
        self.feat_embeddings = nn.Embedding(self.num_features, self.dim)
        self.layers = nn.Sequential(
            nn.Conv2d(self.dim, self.dim * 2, (self.num_feature_groups,
                                               self.window_size), padding=self.window_size // 2),
            nn.MaxPool2d((1, 2))
        )

    def forward(self, feat_matrix, pos_to_predict):
        feat_emb = self.feat_embeddings(feat_matrix).transpose(1, 3)
        # Set positions to predict to zero.
        bs, _, _ = feat_matrix.shape
        batch_i = get_range(bs, 1, 0)
        feat_emb[batch_i, pos_to_predict] = 0.0
        # Run through cnns.
        output = self.layers(feat_emb)
        h, _ = output.max(dim=-1)
        h = h.reshape(bs, -1)
        return h


class Predictor(nn.Module):

    def __init__(self, num_features, dim):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Linear(self.dim * 6, self.dim),
            nn.LeakyReLU(0.1),
            nn.Linear(self.dim, self.num_features),
        )

    def forward(self, h):
        return torch.sigmoid(self.layers(h)).clamp(min=1e-8).log()


@init_g_attr
class Model(nn.Module):

    add_argument('num_features', default=10, dtype=int, msg='total number of phonetic features')
    add_argument('num_feature_groups', default=10, dtype=int, msg='total number of phonetic feature groups')
    add_argument('dim', default=5, dtype=int, msg='dimensionality of feature embeddings and number of hidden units')

    def __init__(self, num_features, num_feature_groups, dim, window_size):
        super().__init__()
        self.encoder = Encoder(num_features, num_feature_groups, dim, window_size)
        self.predictor = Predictor(num_features, dim)

    def forward(self, batch):
        """
        First encode the `feat_matrix` into a vector `h`, then based on it predict the distributions of features.
        """
        h = self.encoder(batch.feat_matrix, batch.pos_to_predict)
        distr = self.predictor(h)
        return distr
