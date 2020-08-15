"""Define classes that deal with IPA symbols."""

from __future__ import annotations

import logging
from collections.abc import Sequence as SequenceABC
from copy import deepcopy
from functools import wraps
from typing import ClassVar, Dict, List, NewType, Optional, TypeVar, Union

import torch
from ipapy import UNICODE_TO_IPA
from ipapy.ipachar import (DG_C_MANNER, DG_C_PLACE, DG_C_VOICING,
                           DG_DIACRITICS, DG_S_BREAK, DG_S_LENGTH, DG_S_STRESS,
                           DG_T_CONTOUR, DG_T_GLOBAL, DG_T_LEVEL, DG_TYPES,
                           DG_V_BACKNESS, DG_V_HEIGHT, DG_V_ROUNDNESS, IPAChar)
from ipapy.ipastring import IPAString
from typeguard import typechecked

from dev_misc import LT
from dev_misc.utils import cached_property
from xib.ipa import Category

Lang = NewType('Lang', str)

# Remove some IPA symbols that are co-articulated.
to_remove_rules_common = {
    'n͡m': ['n', 'm'],
    'd͡b': ['d', 'b'],
    'k͡p': ['k', 'p'],
    't͡p': ['t', 'p']
}
to_remove_rules_pgm = {
    't͡s': ['t', 's'],
    'd͡z': ['d', 'z']
}
to_remove_rules_non = {
    't͡s': ['t', 's'],
}
to_remove_rules_heb = {
    't͡ʃ': ['t', 'ʃ'],
}
to_remove_rules_ara = {
    't͡ʃ': ['t', 'ʃ'],
    't͡s': ['t', 's'],
}
to_remove_rules_lang = {
    'pgm': to_remove_rules_pgm,
    'non': to_remove_rules_non,
    'heb': to_remove_rules_heb,
    'ara': to_remove_rules_ara
}
ipa_chars_to_remove = dict()


class DigraphProcessor:

    def __init__(self, lang: Lang):
        if lang == 'germ':
            lang = 'pgm'
        self.lang = lang

    @cached_property(in_class=True, key=lambda self: self.lang)
    def ipa_chars_to_remove(self) -> Dict[str, List[str]]:
        rules = deepcopy(to_remove_rules_common)
        if self.lang not in to_remove_rules_lang:
            logging.imp(f'Language {self.lang} does not have a language-specific set of digraphs to remove.')
        else:
            rules.update(to_remove_rules_lang[self.lang])
        return rules


class RemovedIpaSymbol(Exception):
    """Raise this if an IPA symbol should be removed."""


class InvalidIpaSymbol(Exception):
    """Raise this if an IPA symbol is invalid."""


def indexify_ipa(col: str, attr: str) -> int:
    cat_cls = Category.get_enum(col)
    return getattr(cat_cls, attr.replace('-', '_').upper()).value.g_idx


col_names = [f'{feat.name.lower()}' for feat in Category]

name2dg = {
    'ptype': DG_TYPES,
    'c_voicing': DG_C_VOICING,
    'c_place': DG_C_PLACE,
    'c_manner': DG_C_MANNER,
    'v_height': DG_V_HEIGHT,
    'v_backness': DG_V_BACKNESS,
    'v_roundness': DG_V_ROUNDNESS,
    'diacritics': DG_DIACRITICS,
    's_stress': DG_S_STRESS,
    's_length': DG_S_LENGTH,
    's_break': DG_S_BREAK,
    't_level': DG_T_LEVEL,
    't_contour': DG_T_CONTOUR,
    't_global': DG_T_GLOBAL
}


def get_fv(dg_values):
    fv = [indexify_ipa(col, dg_values[col]) for col in col_names]
    return torch.LongTensor(fv)


class IpaSingleChar:

    """An instance of this class represents on one single IPA symbol.

    The actual length for its string representation might be more than one-character long. For instance, "t͡s" is three-character long.
    Equality check is based on its string representation.
    """

    @typechecked
    def __init__(self, lang: Lang, char: IPAChar):
        digraph_proc = DigraphProcessor(lang)
        if str(char) in digraph_proc.ipa_chars_to_remove:  # pylint: disable=unsupported-membership-test
            raise RemovedIpaSymbol(f'IPA symbol "{char}" should be removed.')
        self._char = char

    def __str__(self):
        return str(self._char)

    def __repr__(self):
        return f'IpaSingleChar("{str(self)}")'

    @typechecked
    def __eq__(self, other: IpaSingleChar):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    @classmethod
    @typechecked
    def from_str(cls, lang: Lang, s: str) -> IpaSingleChar:
        """Turn a unicode string into an IpaSingleChar object."""
        try:
            return cls(lang, UNICODE_TO_IPA[s])
        except KeyError:
            raise InvalidIpaSymbol(f'Invalid IPA symbol "{s}" for creating an IpaSingleChar instance.')

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            char = super().__getattribute__('_char')
            return getattr(char, attr)

    @property
    def is_cv(self) -> bool:
        return self.is_consonant or self.is_vowel

    @cached_property(in_class=True, key=__str__)
    def dg_values(self) -> Dict[str, str]:

        def use_none_str(x):
            if x is None:
                return 'NONE'
            return x

        ret = dict()
        for col in col_names:
            dg = name2dg[col]
            v = use_none_str(self.dg_value(dg))
            ret[col] = v
        return ret

    @cached_property(in_class=True, key=__str__)
    def feature_vector(self) -> LT:
        return get_fv(self.dg_values)


class IpaUnit:

    """An IpaUnit can consist of more than one IpaSingleChar and forms a basic element in the character set.

    Depending on the definition, long vowels and their short versions can be the same units or not.
    Validation for IpaUnit instances should not take place here -- it should be the responsibility for a segmenter.
    """

    def __init__(self, single_chars: List[IpaSingleChar]):
        self.single_chars = single_chars

    @typechecked
    def __eq__(self, other: IpaUnit):
        return len(self.single_chars) == len(other.single_chars) and all(x == y for x, y in zip(self.single_chars, other.single_chars))

    @classmethod
    @typechecked
    def from_str(cls, lang: Lang, s: str) -> IpaUnit:
        return IpaUnit([IpaSingleChar.from_str(lang, ss) for ss in s])

    def __str__(self):
        return ''.join(str(sc) for sc in self.single_chars)

    def __hash__(self):
        return hash(str(self))

    def __repr__(self):
        return f'IpaUnit("{str(self)}")'

    @cached_property(in_class=True, key=__str__)
    def dg_values(self) -> Dict[str, str]:
        """`dg_values` for a unit is the combination of its constituing characters in a bag-of-words-like way. This would ignore order in the order of first-come-first-serve."""
        ret = {k: 'NONE' for k in col_names}
        for sc in self.single_chars:
            for k, v in sc.dg_values.items():
                if v is not 'NONE' and ret[k] == 'NONE':
                    ret[k] = v
        return ret

    @cached_property(in_class=True, key=__str__)
    def feature_vector(self) -> LT:
        return get_fv(self.dg_values)


class InvalidIpaTranscription(Exception):
    """Raise this if the provided IPA transcription contains some invalid patterns."""


class IpaUnitSegmenter:

    """Used for segmenting a raw string into a list of IpaUnit."""

    def segment(self,
                lang: Lang,
                raw_string: str,
                use_length: bool = False) -> List[IpaUnit]:

        # First pass to get the IPAChar instances.
        ipa_chars = IPAString(unicode_string=raw_string).ipa_chars

        # Second pass to convert them into IpaSingleChar instances. Pay special attention to removed characters.
        sc_lst = list()
        digraph_proc = DigraphProcessor(lang)
        for char in ipa_chars:
            try:
                sc = IpaSingleChar(lang, char)
                sc_lst.append(sc)
            except RemovedIpaSymbol:
                for v in digraph_proc.ipa_chars_to_remove[str(char)]:  # pylint: disable=unsubscriptable-object
                    sc_lst.append(IpaSingleChar(lang, UNICODE_TO_IPA[v]))

        # Merge relevant single characters to form units.
        ret = list()
        i = 0
        while i < len(sc_lst):
            sc = sc_lst[i]
            if not sc.is_cv:
                raise InvalidIpaTranscription(
                    f'Every single character should start with a consonant or a vowel, but got "{sc}" at position {i} for input string {raw_string}.')

            j = i + 1
            while j < len(sc_lst) and not sc_lst[j].is_cv:
                j += 1

            if use_length:
                unit = IpaUnit(sc_lst[i: j])
            else:
                unit = IpaUnit([sc for sc in sc_lst[i: j] if str(sc) != 'ː'])
            ret.append(unit)
            i = j

        return ret


class IpaSequence(SequenceABC):

    """A sequence of units."""

    def __init__(self,
                 lang: Lang,
                 raw_string: Optional[str] = None,
                 *,
                 units: Optional[List[IpaUnit]] = None,
                 segmenter_kwargs=None):
        self.lang = lang
        if units is not None:
            self.units = units
        else:
            segmenter = IpaUnitSegmenter()
            segmenter_kwargs = segmenter_kwargs or dict()
            units = segmenter.segment(lang, raw_string, **segmenter_kwargs)
            self.units = units

    @cached_property
    def feature_matrix(self) -> LT:
        fm = list()
        for unit in self.units:
            fv = unit.feature_vector
            fm.append(fv)
        return torch.stack(fm, dim=0)

    def __len__(self):
        return len(self.units)

    @typechecked
    def __getitem__(self, idx: Union[int, slice]) -> IpaSequence:
        if isinstance(idx, int):
            units = [self.units[idx]]
        else:
            units = self.units[idx]
        return IpaSequence(lang, units=units)

    def __str__(self):
        return ''.join(map(str, self.units))

    def __repr__(self):
        return f'IpaSequence("{self}")'

    @typechecked
    def __eq__(self, other: IpaSequence):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))


Content = TypeVar('Content', str, IpaSequence)
