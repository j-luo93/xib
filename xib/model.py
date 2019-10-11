import torch
import torch.nn as nn
from arglib import add_argument, init_g_attr
from devlib import get_range
from xib.cfg import Category, conditions, no_none_predictions


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

        # feat_emb = self.feat_embeddings(feat_matrix).transpose(1, 3)
        # # Set positions to predict to zero.
        # bs, _, _ = feat_matrix.shape
        # batch_i = get_range(bs, 1, 0)
        # # feat_emb[batch_i, :, :, pos_to_predict] = 0.0
        # # Run through cnns.
        # output = self.layers(feat_emb)
        # h, _ = output.max(dim=-1)
        # h = h.reshape(bs, -1)
        # return h


@init_g_attr(default='property')
class Predictor(nn.Module):

    def __init__(self, num_features, hidden_size, window_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
        )
        self.feat_predictors = nn.ModuleDict()
        for name, cat in Category.get_named_cat_enums():
            self.feat_predictors[name] = nn.Linear(hidden_size, len(cat))

    def forward(self, h):
        shared_h = self.layers(h)
        ret = dict()
        for name, layer in self.feat_predictors.items():
            out = layer(shared_h)
            if name in no_none_predictions:
                index = no_none_predictions[name]
                out[:, index.f_idx] = -999.9
            ret[name] = torch.log_softmax(out, dim=-1)
        # Deal with conditions for some categories
        for name, index in conditions.items():
            # Find out the exact value to be conditioned on.
            condition_name = Category(index.c_idx).name
            condition_idx = index.f_idx
            condition_log_probs = ret[condition_name][:, condition_idx]
            ret[name] = ret[name] + condition_log_probs.unsqueeze(dim=-1)

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

    def forward(self, batch):
        """
        First encode the `feat_matrix` into a vector `h`, then based on it predict the distributions of features.
        """
        h = self.encoder(batch.feat_matrix, batch.pos_to_predict)
        distr = self.predictor(h)
        return distr
