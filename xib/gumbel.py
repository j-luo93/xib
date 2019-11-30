"""
Modified from https://blog.evjang.com/2016/11/tutorial-categorical-variational.html.
"""

from dev_misc.devlib.named_tensor import get_named_range
from typing import Optional, Sequence, Tuple

import torch

from dev_misc import FT, LT, get_tensor
from dev_misc.devlib.named_tensor import NoName


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
    logits = logits.align_to('batch', 'length', 'label')
    y = gumbel_softmax_sample(logits, temperature, num_samples)
    y = y.align_to('batch', 'length', 'label', ...)
    max_values, max_inds = y.max(dim='label')
    y_one_hot = (max_values.align_as(y) == y).float()
    y_one_hot = (y_one_hot - y).detach() + y
    bi = get_named_range(logits.size('batch'), 'batch').align_as(max_inds)
    li = get_named_range(logits.size('length'), 'length').align_as(max_inds)
    if num_samples is None:
        with NoName(max_inds, y_one_hot, bi, li):
            probs = y_one_hot[bi, li, max_inds]
        probs.rename_('batch', 'length')
    else:
        si = get_named_range(max_inds.size('sample'), 'sample').align_as(max_inds)
        with NoName(max_inds, y_one_hot, bi, li, si):
            probs = y_one_hot[bi, li, max_inds, si]
        probs.rename_('batch', 'length', 'sample')
    seq_probs = (1e-8 + probs).log().sum(dim='length').exp()

    return y, y_one_hot, max_inds, seq_probs
