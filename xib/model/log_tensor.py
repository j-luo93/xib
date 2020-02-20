from __future__ import annotations

import math
from functools import wraps
from typing import Callable, Iterable, Optional, Tuple, Union

import numpy as np
import torch

from dev_misc import FT, LT
from dev_misc.devlib.named_tensor import patch_named_tensors
from dev_misc.utils import cached_property, check_explicit_arg

EPS = 1e-16


def th_logsumexp(a: FT, dim: Union[str, int] = None, b: Optional[FT] = None, keepdims: bool = False, return_sign: bool = False):
    """This follows the scipy implementation here:
    https://github.com/scipy/scipy/blob/29dbc361056df93fe659dcb7362fa2c0e647a14b/scipy/special/_logsumexp.py#L8-L127
    """
    if b is not None:
        b_is_zero = (b == 0)
        if b_is_zero.any():
            a = a - torch.where(b_is_zero, torch.full_like(b, np.inf), torch.zeros_like(b))

    a_max, _ = a.max(dim=dim, keepdims=True)

    max_is_finite = torch.isfinite(a_max)
    if a_max.ndim > 0:
        a_max = torch.where(max_is_finite, a_max, torch.zeros_like(a_max))
    elif not max_is_finite:
        a_max = 0

    if b is not None:
        tmp = b * (a - a_max).exp()
    else:
        tmp = (a - a_max).exp()

    s = tmp.sum(dim=dim, keepdims=keepdims)
    if return_sign:
        sgn = s.sign()
        s *= sgn
    out = (s + EPS).log()

    if not keepdims:
        a_max = a_max.squeeze(dim=dim)
    out = out + a_max

    if return_sign:
        return out, sgn
    else:
        return out


def _check_log_tensor_type(tensor):
    if not isinstance(tensor, LogTensor):
        raise TypeError(f'Expect a LogTensor instance but got {type(tensor)}.')


def _inherit(func_name: str) -> Callable:

    def wrapped(self, *args, **kwargs) -> LogTensor:
        sign = getattr(self.sign, func_name)(*args, **kwargs)
        storage = getattr(self.storage, func_name)(*args, **kwargs)
        return LogTensor(sign, storage)

    return wrapped


def _do_as(func_name: str) -> Callable:

    def wrapped(self, other: LogTensor) -> LogTensor:
        sign = getattr(self.sign, func_name)(other.sign)
        storage = getattr(self.storage, func_name)(other.storage)
        return LogTensor(sign, storage)

    return wrapped


class LogTensor:

    def __init__(self, sign: FT, storage: FT, value: Optional[FT] = None):
        self._sign = sign
        self._storage = storage
        self._value = value

    def has_names(self) -> bool:
        return self.storage.has_names()

    @classmethod
    def from_torch(cls, tensor: FT, log_scale: bool = False) -> LogTensor:
        if log_scale:
            value = None
            sign = torch.ones_like(tensor)
            storage = tensor
        else:
            value = tensor
            sign = tensor.sign()
            storage = (tensor.abs() + EPS).log()
        return LogTensor(sign, storage, value=value)

    @property
    def storage(self) -> FT:
        return self._storage

    @property
    def sign(self) -> FT:
        return self._sign

    @cached_property
    def value(self) -> FT:
        if self._value is None:
            return self._sign * self._storage.exp()
        else:
            return self._value

    def __repr__(self):
        return repr(self.value)

    def __mul__(self, other: Union[LogTensor, int, float]) -> LogTensor:
        if isinstance(other, (int, float)):
            sign = self.sign
            if other < 0:
                sign = -self.sign
                other = -other
            storage = self.storage + math.log(other)
        else:
            _check_log_tensor_type(other)
            sign = self.sign * other.sign
            storage = self.storage + other.storage
        return LogTensor(sign, storage)

    def __truediv__(self, other: Union[LogTensor, float, int]) -> LogTensor:
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        else:
            mul_other = LogTensor(other.sign, -other.storage)
            return self * mul_other

    def __rmul__(self, other: Union[int, float]) -> LogTensor:
        return self * other

    def __add__(self, other: LogTensor) -> LogTensor:
        _check_log_tensor_type(other)
        a = torch.stack([self.storage, other.storage], new_name='____')
        b = torch.stack([self.sign, other.sign], new_name='____')
        storage, sign = th_logsumexp(a, b=b, dim='____', return_sign=True)
        return LogTensor(sign, storage)

    align_to = _inherit('align_to')
    align_as = _do_as('align_as')
    expand = _inherit('expand')
    expand_as = _do_as('expand_as')
    __getitem__ = _inherit('__getitem__')
    rename = _inherit('rename')
    rename_ = _inherit('rename_')
    chunk = _inherit('chunk')

    reveal_names = _inherit('reveal_names')
    hide_names = _inherit('hide_names')

    @classmethod
    def stack(cls, lst: Iterable[LogTensor], new_name: str = None) -> LogTensor:
        check_explicit_arg(new_name)

        sign_lst = [lt.sign for lt in lst]
        storage_lst = [lt.storage for lt in lst]
        sign = torch.stack(sign_lst, new_name=new_name)
        storage = torch.stack(storage_lst, new_name=new_name)
        return LogTensor(sign, storage)

    @classmethod
    def cat(cls, lst: Iterable[LogTensor], dim: str) -> LogTensor:
        sign_lst = [lt.sign for lt in lst]
        storage_lst = [lt.storage for lt in lst]
        sign = torch.cat(sign_lst, dim=dim)
        storage = torch.cat(storage_lst, dim=dim)
        return LogTensor(sign, storage)

    def sum(self, dim: Optional[str, int]) -> LogTensor:
        storage, sign = th_logsumexp(self.storage, b=self.sign, dim=dim, return_sign=True)
        return LogTensor(sign, storage)

    @classmethod
    def sum_all(self, lst: Iterable[LogTensor]) -> LogTensor:
        """Sum all the LogTensor instances in `lst`."""
        return LogTensor.stack(lst, new_name='____').sum(dim='____')

    def max(self, dim: Union[str, int]) -> Tuple[LogTensor, LT]:
        is_pos = self.sign == 1.0
        has_pos = is_pos.any(dim=dim, keepdims=True)
        value = self.storage
        pos_part = value - torch.where(is_pos, torch.zeros_like(value), torch.full_like(value, np.inf))
        neg_part = -value
        max_v, max_i = torch.where(has_pos, pos_part, neg_part).max(dim=dim)
        has_pos = has_pos.squeeze(dim=dim)
        max_v = torch.where(has_pos, max_v, -max_v)

        ones = torch.ones_like(max_v)
        sign = torch.where(has_pos, ones, -ones)
        return LogTensor(sign, max_v), max_i

    @classmethod
    def max_all(self, lst: Iterable[LogTensor]) -> LogTensor:
        return LogTensor.stack(lst, new_name='____').max(dim='____')

    def size(self, *args, **kwargs):
        return self.storage.size(*args, **kwargs)
