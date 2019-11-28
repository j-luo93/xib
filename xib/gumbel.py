"""
Modified from https://blog.evjang.com/2016/11/tutorial-categorical-variational.html.
"""

from typing import Optional
from typing import Sequence, Tuple

import torch

from dev_misc import FT, LT

from dev_misc import get_tensor


def sample_gumbel(shape: Sequence[int], eps: float = 1e-20) -> FT:
    """Sample from Gumbel(0, 1)"""
    U = get_tensor(torch.rand(shape))
    return -(-(U + eps).log() + eps).log()


def gumbel_softmax_sample(logits: FT, temperature: float, num_samples: Optional[int] = None) -> FT:
    """Draw a sample from the Gumbel-Softmax distribution"""
    new_names = logits.names
    shape = tuple(logits.shape)
    if num_samples is not None:
        new_names += ('sample', )
        shape += (num_samples, )
    noise = sample_gumbel(shape).rename(*new_names)
    y = logits.align_as(noise) + noise
    return (y / temperature).log_softmax(dim='label').exp()


def gumbel_softmax(logits: FT, temperature: float, num_samples: Optional[int] = None) -> Tuple[FT, FT, LT]:
    """Sample from the Gumbel-Softmax distribution and optionally discretize."""
    y = gumbel_softmax_sample(logits, temperature, num_samples)
    max_values, max_inds = y.max(dim='label')
    y_one_hot = (max_values.align_as(y) == y).float()
    y_one_hot = (y_one_hot - y).detach() + y
    return y, y_one_hot, max_inds
