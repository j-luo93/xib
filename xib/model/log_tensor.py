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

# profile = lambda x: x


# @profile
def th_logsumexp(a: FT, b: FT, dim: Union[str, int] = None, keepdims: bool = False):
    """This follows the scipy implementation here:
    https://github.com/scipy/scipy/blob/29dbc361056df93fe659dcb7362fa2c0e647a14b/scipy/special/_logsumexp.py#L8-L127
    """
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
    sgn = s.sign()
    s *= sgn
    out = (s + EPS).log()

    if not keepdims:
        a_max = a_max.squeeze(dim=dim)
    out = out + a_max

    return out, sgn


def _check_log_tensor_type(tensor):
    if not isinstance(tensor, LogTensor):
        raise TypeError(f'Expect a LogTensor instance but got {type(tensor)}.')


def _inherit(func_name: str) -> Callable:

    def wrapped(self, *args, **kwargs) -> LogTensor:
        storage = getattr(self.storage, func_name)(*args, **kwargs)
        if self.sign is None:
            sign = None
        else:
            sign = getattr(self.sign, func_name)(*args, **kwargs)
        return LogTensor(storage, sign)

    return wrapped


def _do_as(func_name: str) -> Callable:

    def wrapped(self, other: LogTensor) -> LogTensor:
        storage = getattr(self.storage, func_name)(other.storage)
        if self.sign is None:
            sign = None
        else:
            sign = getattr(self.sign, func_name)(other.storage)
        return LogTensor(storage, sign)

    return wrapped


def _get_sign_lst(sign_lst, storage_lst):
    if any(sign is not None for sign in sign_lst):
        sign_lst = [torch.ones_like(storage) if sign is None else sign
                    for sign, storage in zip(sign_lst, storage_lst)]
        return sign_lst
    else:
        return None


def mul_sign(s1: Union[None, FT], s2: Union[None, FT]) -> Union[None, FT]:
    if s1 is None:
        ret = s2
    elif s2 is None:
        ret = s1
    else:
        ret = s1 * s2
    return ret


class LogTensor:

    def __init__(self, storage: FT, sign: Optional[FT] = None, value: Optional[FT] = None):
        self._storage = storage
        self._sign = sign
        self._value = value

    def has_names(self) -> bool:
        return self.storage.has_names()

    @classmethod
    def from_torch(cls, tensor: FT, log_scale: bool = False, nonneg: bool = False) -> LogTensor:
        if log_scale:
            value = None
            sign = torch.ones_like(tensor)
            storage = tensor
        else:
            value = tensor
            sign = tensor.sign()
            storage = (tensor.abs() + EPS).log()

        if nonneg or log_scale:
            sign = None
        return LogTensor(storage, sign=sign, value=value)

    @property
    def storage(self) -> FT:
        return self._storage

    @property
    def sign(self) -> FT:
        return self._sign

    @cached_property
    def value(self) -> FT:
        if self._value is None:
            ret = self._storage.exp()
            if self._sign is not None:
                ret = self._sign * ret
            return ret
        else:
            return self._value

    def __repr__(self):
        return repr(self.value)

    def __mul__(self, other: Union[LogTensor, int, float]) -> LogTensor:
        if isinstance(other, (int, float)):
            sign = self.sign
            if other < 0:
                sign = -self.sign if self.sign is not None else None
                other = -other
            storage = self.storage + math.log(other)
        else:
            _check_log_tensor_type(other)
            sign = mul_sign(self.sign, other.sign)
            storage = self.storage + other.storage
        return LogTensor(storage, sign)

    def __truediv__(self, other: Union[LogTensor, float, int]) -> LogTensor:
        if isinstance(other, (int, float)):
            return self * (1.0 / other)
        else:
            mul_other = LogTensor(-other.storage, other.sign)
            return self * mul_other

    def __rmul__(self, other: Union[int, float]) -> LogTensor:
        return self * other

    def __add__(self, other: LogTensor) -> LogTensor:
        _check_log_tensor_type(other)
        a = torch.stack([self.storage, other.storage], new_name='____')
        if self.sign is None and other.sign is None:
            storage = a.logsumexp(dim='____')
            return LogTensor(storage)
        else:
            if self.sign is not None and other.sign is not None:
                b = torch.stack([self.sign, other.sign], new_name='____')
            elif self.sign is None:
                b = torch.stack([torch.ones_like(self.storage), other.sign], new_name='____')
            else:
                b = torch.stack([self.sign, torch.ones_like(other.storage)], new_name='____')
            storage, sign = th_logsumexp(a, b, dim='____')
            return LogTensor(storage, sign)

    def __neg__(self) -> LogTensor:
        if self.sign is None:
            sign = -torch.ones_like(self.storage)
        else:
            sign = -self.sign
        return LogTensor(self.storage, sign)

    def __sub__(self, other: LogTensor) -> LogTensor:
        _check_log_tensor_type(other)
        return self + (-other)

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

    # @profile
    @classmethod
    def stack(cls, lst: Iterable[LogTensor], new_name: str = None) -> LogTensor:
        check_explicit_arg(new_name)
        storage_lst = [lt.storage for lt in lst]
        storage = torch.stack(storage_lst, new_name=new_name)
        sign_lst = [lt.sign for lt in lst]
        sign_lst = _get_sign_lst(sign_lst, storage_lst)
        if sign_lst:
            sign = torch.stack(sign_lst, new_name=new_name)
        else:
            sign = None
        return LogTensor(storage, sign)

    @classmethod
    def cat(cls, lst: Iterable[LogTensor], dim: str) -> LogTensor:
        storage_lst = [lt.storage for lt in lst]
        storage = torch.cat(storage_lst, dim=dim)
        sign_lst = [lt.sign for lt in lst]
        sign_lst = _get_sign_lst(sign_lst, storage_lst)
        if sign_lst:
            sign = torch.cat(sign_lst, dim=dim)
        else:
            sign = None
        return LogTensor(storage, sign)

    def sum(self, dim: Optional[str, int]) -> LogTensor:
        if self.sign is None:
            storage = self.storage.logsumexp(dim=dim)
            return LogTensor(storage)
        else:
            storage, sign = th_logsumexp(self.storage, self.sign, dim=dim)
            return LogTensor(storage, sign)

    @classmethod
    def sum_all(self, lst: Iterable[LogTensor]) -> LogTensor:
        """Sum all the LogTensor instances in `lst`."""
        return LogTensor.stack(lst, new_name='____').sum(dim='____')

    def max(self, dim: Union[str, int]) -> Tuple[LogTensor, LT]:
        if self.sign is None:
            max_v, max_i = self.storage.max(dim=dim)
            return LogTensor(max_v), max_i

        # The result should has max sign, and within all values with max sign, has the biggest/smallest storage depending on the sign value.
        max_sign, _ = self.sign.max(dim=dim, keepdims=True)
        v = self.storage
        masked_v = v - torch.where(self.sign == max_sign, torch.zeros_like(v), torch.full_like(v, np.inf))
        neg_only = max_sign == -1.0
        max_v, max_i = torch.where(neg_only, -masked_v, masked_v).max(dim=dim)
        max_v = torch.where(neg_only.squeeze(dim=dim), -max_v, max_v)
        return LogTensor(max_v, max_sign.squeeze(dim=dim)), max_i

        # # Old code here.
        # is_pos = self.sign == 1.0
        # has_pos = is_pos.any(dim=dim, keepdims=True)
        # value = self.storage
        # pos_part = value - torch.where(is_pos, torch.zeros_like(value), torch.full_like(value, np.inf))
        # neg_part = -value
        # max_v, max_i = torch.where(has_pos, pos_part, neg_part).max(dim=dim)
        # has_pos = has_pos.squeeze(dim=dim)
        # max_v = torch.where(has_pos, max_v, -max_v)

        # ones = torch.ones_like(max_v)
        # sign = torch.where(has_pos, ones, -ones)
        # return LogTensor(max_v, sign), max_i

    @classmethod
    def max_all(self, lst: Iterable[LogTensor]) -> LogTensor:
        return LogTensor.stack(lst, new_name='____').max(dim='____')

    def size(self, *args, **kwargs):
        return self.storage.size(*args, **kwargs)
