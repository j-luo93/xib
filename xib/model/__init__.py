import torch

from dev_misc.arglib import add_argument

# IDEA(j_luo) typing!
LT = torch.LongTensor
FT = torch.FloatTensor
BT = torch.BoolTensor

add_argument('num_features', default=10, dtype=int, msg='total number of phonetic features')
add_argument('num_feature_groups', default=10, dtype=int, msg='total number of phonetic feature groups')
add_argument('dim', default=5, dtype=int, msg='dimensionality of feature embeddings')
add_argument('hidden_size', default=5, dtype=int, msg='hidden size')
