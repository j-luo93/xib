import torch
import torch.nn as nn

from arglib import add_argument, init_g_attr
from devlib import get_range


class Encoder(nn.Module):

    def __init__(self, num_features, dim):
        super().__init__()
        self.dim = dim
        self.num_features = num_features
        self.feat_embeddings = nn.Embedding(self.num_features, self.dim)
        self.layers = nn.Sequential(
            nn.Conv1d(1, self.dim, 3),
            nn.MaxPool1d(2),
            nn.Conv1d(self.dim, self.dim // 2, 3),
            nn.MaxPool1d(2),
            nn.Conv1d(self.dim // 2, self.dim, 3),
        )

    def forward(self, ipa_matrix, pos_to_predict):
        ipa_emb = self.feat_embeddings(ipa_matrix)
        # Set positions to predict to zero.
        bs, ws = ipa_matrix.shape
        batch_i = get_range(bs, 2, 0)
        window_i = get_range(bs, 2, 1)
        ipa_emb[batch_i, window_i, pos_to_predict] = 0.0
        # Run through cnns.
        output = self.layers(ipa_emb)
        h, _ = output.max(dim=1)
        return h


class Predictor(nn.Module):

    def __init__(self, num_features, dim):
        super().__init__()
        self.num_features = num_features
        self.dim = dim
        self.layers = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LeakyReLU(0.1)
            nn.Linear(self.dim, self.num_features),
        )

    def forward(self, h):
        return self.layers(h)


@init_g_attr
class Model(nn.Module):

    add_argument('num_features', dtype=10, dtype=int, msg='total number of phonetic features')
    add_argument('dim', default=5, dtype=int, msg='dimensionality of feature embeddings and number of hidden units')

    def __init__(self, num_features, dim):
        super().__init__()
        self.encoder = Encoder(dim, )
        self.predictor = Predictor()

    def forward(self, batch):
        """
        First encode the `ipa_matrix` into a vector `h`, then based on it predict the distributions of features.
        """
        h = self.encoder(batch.ipa_matrix, batch.pos_to_predict)
        distr = self.predictor(h)
        return distr
