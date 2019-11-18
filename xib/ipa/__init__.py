"""
Define the common interface for old style and new style ipa categories.
"""

from typing import List, Tuple, Union

import inflection

from dev_misc.arglib import g
from dev_misc.utils import check_explicit_arg

from .ipa import (Category, Index, IPAFeature, conditions, get_enum_by_cat,
                  no_none_predictions)
from .ipax import CategoryX, DistEnum


def should_predict_none(cat_name, *, new_style: bool = None) -> bool:
    check_explicit_arg(new_style)
    if new_style:
        cat_name = cat_name.strip('_X')
    return cat_name not in no_none_predictions


def get_none_index(cat_name: str, *, new_style: bool = None) -> int:
    """Return the feature index for the NONE value of a certain category. Note that for new style ipa, complex feature groups like CPlaceX and CMannerX do not have the feature index."""
    check_explicit_arg(new_style)
    if new_style:
        if cat_name in ['CMannerX', 'CPlaceX']:
            raise ValueError(f'Cannot return feature index for NONE value of {cat_name}')
        e = CategoryX.get_enum(cat_name)
        return e['NONE']
    else:
        e = Category.get_enum(cat_name)
        return e['NONE'].value.f_idx


Cat = Union[Category, CategoryX]
Enum = Union[DistEnum, IPAFeature]


def should_include(groups: str, cat: Union[Cat, 'Name']) -> bool:
    try:
        name = cat.name
    except AttributeError:
        name = cat.value
    if name.startswith('P') and 'p' in groups:
        return True
    if name.startswith('C') and 'c' in groups:
        return True
    if name.startswith('V') and 'v' in groups:
        return True
    if name.startswith('D') and 'd' in groups:
        return True
    if name.startswith('S') and 's' in groups:
        return True
    if name.startswith('T') and 't' in groups:
        return True
    return False


def get_needed_categories(groups, *, new_style: bool = None, breakdown: bool = None) -> List[Enum]:
    """
    Get all categories (actually their enum classes) that are specified by the groups argument.
    if `breakdown` is True, then a complex feature group will be broken down into its parts.
    """
    check_explicit_arg(new_style)
    check_explicit_arg(breakdown)
    if breakdown and not new_style:
        raise ValueError(f'Conflicting argument values for breakdown "{breakdown}" and "{new_style}".')

    cat_cls = CategoryX if new_style else Category
    ret = list()
    for cat in cat_cls:
        if should_include(groups, cat):
            e = cat_cls.get_enum(cat.name)
            if breakdown and e.num_groups() > 1:
                ret.extend(e.parts())
            else:
                ret.append(e)
    return ret


class Name:
    """A Name instance that takes care of all different forms of str."""

    def __init__(self, name: str, fmt: str):
        if fmt not in ['snake', 'camel']:
            raise ValueError(f'fmt can only be "snake" or "camel".')
        self._fmt = fmt
        self._name = name

    def __hash__(self):
        return hash(self.canonicalize().value)

    def __eq__(self, other):
        return self.canonicalize().value == other.canonicalize().value

    def canonicalize(self) -> 'Name':
        if self._fmt == 'camel':
            return self
        else:
            return self.lowercase.camel

    @property
    def value(self):
        return self._name

    @property
    def lowercase(self) -> 'Name':
        if self._fmt == 'camel':
            raise RuntimeError(f'Cannot lowercase the name in camel case.')
        new_name = self._name.lower()
        return Name(new_name, self._fmt)

    @property
    def capital(self) -> 'Name':
        if self._fmt == 'camel':
            raise RuntimeError(f'Cannot uppercase the name in camel case.')
        new_name = self._name.upper()
        return Name(new_name, self._fmt)

    @property
    def camel(self) -> 'Name':
        new_name = inflection.camelize(self._name)
        return Name(new_name, 'camel')

    @property
    def snake(self) -> 'Name':
        new_name = inflection.underscore(self._name)
        return Name(new_name, 'snake')

    def __repr__(self):
        return f'Name("{self._name}", fmt={self._fmt})'

    def __str__(self):
        return self._name


def get_index(name: Name, *, new_style: bool = None) -> int:
    """Given a Name instance, return the category index in the old style."""
    check_explicit_arg(new_style)
    if new_style:
        name = name.snake.capital.value.strip('_X')
    else:
        name = name.snake.capital.value
    return Category[name].value


def get_new_style_enum(c_idx: int) -> DistEnum:
    """Given c_idx in the old style, return the corresponding enum class in the new style."""
    old_cat = Category(c_idx)
    new_enum = CategoryX.get_enum(old_cat.name + '_X')
    return new_enum
