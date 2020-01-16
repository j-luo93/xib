# pylint: disable=assigning-non-slot,no-member,unsupported-assignment-operation,unsubscriptable-object

import re
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple

import pandas as pd
import torch.nn.init

from dev_misc.utils import Singleton

RE_HV = 'ƕ'
RE_CHAR = fr'[\wïþÞ{RE_HV}]'
RE_C = fr'[bdfghjklmnpqrstwxzþÞ{RE_HV}'
RE_V = fr'[aeuiïoāăēĕŏōŭū]'
RE_C_BR = fr'[rh{RE_HV}]'
RE_D = fr'[dtþ]'
RE__C = fr'(?={RE_C}+$)'


def undefined(func):

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        assert len(args) == 1 and len(kwargs) == 0
        return args[0]

    return wrapped


class Variable:

    def __init__(self):
        self._single_value = None
        self._value_dict = dict()
        self._undefined = True

    def _check_defined(self):
        if self._undefined:
            raise RuntimeError(f'Undefined variable.')

    @property
    def value(self):
        self._check_defined()
        if not self._value_dict:
            return self._single_value
        return self._value_dict

    def _check_single(self):
        assert not self._value_dict

    def _check_dict(self):
        assert self._single_value is None

    @value.setter
    def value(self, new_value):
        self._check_single()
        self._single_value = new_value
        self._undefined = False

    def __setitem__(self, key, value):
        self._check_dict()
        self._value_dict[key] = value
        self._undefined = False

    def __getitem__(self, key):
        self._check_dict()
        return self._value_dict[key]

    def __delitem__(self, key):
        self._check_dict()
        del self._value_dict[key]

    def __repr__(self):
        self._check_defined()
        return repr(self.value)

    def items(self):
        self._check_defined()
        self._check_dict()
        return self._value_dict.items()

    def update_key(self, key, value):
        self._check_dict()
        if key in self._value_dict:
            keys = [key]
        else:
            keys = [k for k in self._value_dict if key in k]
        for k in keys:
            if value is None:
                del self._value_dict[k]
            else:
                self._value_dict[k] = value


vf = lambda: field(init=False, default_factory=Variable, repr=False)


def gen_sub(sub_rules):
    def sub_func(self, s):
        return sub(s, sub_rules)
    return sub_func

# TODO(j_luo) refactor this
@dataclass
class BaseClass:
    Lemma: str


class MorphClassRegistry(Singleton):

    _id2mc: ClassVar[Dict[int, BaseClass]] = dict()

    def register_dataclass(self, idx: int):
        cls = type(self)
        if idx in cls._id2mc:
            raise ValueError(f'Index {idx} has already been assigned.')

        def decorator(mc_cls):
            mc_dc_cls = dataclass(mc_cls)
            cls._id2mc[idx] = mc_dc_cls
            return mc_cls

        return decorator

    def items(self):
        cls = type(self)
        return cls._id2mc.items()


mcr = MorphClassRegistry()
reg_dc = mcr.register_dataclass


@reg_dc(1)
class Indeclinable(BaseClass):
    pass
    # Form: Variable = vf()

    # def __post_init__(self):
    #     self.Form.value = self.Lemma


def incomplete(func):

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        raise RuntimeError(f'Still missing the treatment of special parameters.')

    return wrapped


@dataclass
class BaseMorphClass(BaseClass):

    # Form: Variable = vf()
    Root: Variable = vf()
    # Suffix: Variable = vf()

    # lemma_specific_suffix_lst: List[str] = field(init=False, repr=False, default=None)

    id2tag: ClassVar[List[str]] = None
    # suffix_lst: ClassVar[List[str]] = None
    # suffix_override: ClassVar[List[Tuple[str, str]]] = None
    # form_remove: ClassVar[List[str]] = None

    def __post_init__(self):
        # self.get_Suffix()
        self.get_Root()
        # self.get_Form()

    # def get_Suffix(self):
    #     cls = type(self)
    #     suffix_lst = self.lemma_specific_suffix_lst or cls.suffix_lst
    #     for tag, suffix in zip(cls.id2tag, suffix_lst):
    #         if suffix is not MISSING:
    #             self.Suffix[tag] = suffix

    #     if cls.suffix_override:
    #         for pattern, new_suf in cls.suffix_override:
    #             self.Suffix.update_key(pattern, new_suf)

    def get_Root(self):
        self.Root.value = self.derive_root(self.Lemma)

    # def get_Form(self):
    #     for tag, suffix in self.Suffix.items():
    #         self.Form[tag] = self.phonology(self.Root.value + suffix)

    #     cls = type(self)
    #     if cls.form_remove:
    #         for tag in cls.form_remove:
    #             self.Form.update_key(tag, None)


# TODO(j_luo) reg this
@dataclass
class Noun(BaseMorphClass):
    Lemma: str
    Pluraletantum: bool = False
    Singularetantum: bool = False

    id2tag = [
        'NOM-SING', 'ACC-SING', 'DAT-SING', 'GEN-SING',
        'VOC-SING', 'NOM-PLUR', 'ACC-PLUR', 'DAT-PLUR',
        'GEN-PLUR', 'VOC-PLUR'
    ]

    @undefined
    def derive_root(self):
        pass

    @undefined
    def derive_root_from_plural(self):
        pass

    # def phonology(self, crude_form: str) -> str:
    #     ret = re.sub(fr'({RE_V})b(s?)$', r'\1f\2', crude_form)
    #     ret = re.sub(fr'({RE_V})d(s?)$', r'\1þ\2', ret)
    #     ret = re.sub(fr'({RE_V})z$', r'\1s', ret)
    #     ret = re.sub(fr'o(?=e?i)', r'au', ret)
    #     ret = re.sub(fr'({RE_C}[ai])w(j|s?$)', r'\1u\2', ret)
    #     ret = re.sub(fr'nn(?=[stþ]$)', r'n', ret)
    #     return ret

    # def get_Form(self):
    #     super().get_Form()
    #     if self.Pluraletantum:
    #         self.Form.update_key('SING', None)
    #     if self.Singularetantum:
    #         self.Form.update_key('PLUR', None)

    def get_Root(self):
        super().get_Root()
        if self.Pluraletantum:
            self.Root.value = self.derive_root_from_plural(self.Lemma)  # pylint: disable=too-many-function-args


@dataclass
class _VocStems(Noun):
    """Vocalic stems."""


MISSING = object()


@dataclass
class _MaStems(_VocStems):

    # suffix_lst = ["s", MISSING, "a", "is", MISSING, "os",
    #               "ans", "am", "e", '']

    def derive_root_from_plural(self, plural: str) -> str:
        return re.sub(r'os$', '', plural)


@dataclass
class _NaStems(_VocStems):

    # suffix_lst = ["s", MISSING, "a", "is", MISSING, "os",
    #               "ans", "am", "e", '']

    def derive_root_from_plural(self, plural: str) -> str:
        return re.sub(r'a$', '', plural)


@reg_dc(3)
class Ma(_MaStems):
    Lemma: str
    z_Assimilation: bool = False
    Auslautverhärtung: bool = True

    def derive_root(self, s: str) -> str:
        return re.sub(r's$', '', s)

    def derive_root_from_plural(self, plural: str) -> str:
        return re.sub(r'os$', '', plural)

    def cancel_final_devoicing(self, s: str) -> str:
        ret = re.sub(fr'({RE_V})f$', r'\1b', s)
        ret = re.sub(fr'({RE_V})þ$', r'\1d', ret)
        ret = re.sub(fr'({RE_V})s$', r'\1z', ret)
        return ret

    def get_Root(self):
        super().get_Root()
        if self.z_Assimilation:
            self.Root.value = self.Lemma
        if self.Auslautverhärtung:
            self.Root.value = self.cancel_final_devoicing(self.Root.value)

    # def get_Form(self):
    #     super().get_Form()
    #     if self.z_Assimilation:
    #         self.Form['NOM-SING'] = self.Lemma


@reg_dc(4)
class Na(_NaStems):
    Lemma: str
    Auslautverhärtung: bool = True

    @undefined
    def derive_root(self):
        pass

    def derive_root_from_plural(self, plural: str):
        return re.sub(r'a$', '', plural)

    def cancel_final_devoicing(self, s):
        ret = re.sub(fr'({RE_V})f$', r'\1b', s)
        ret = re.sub(fr'({RE_V})þ$', r'\1d', ret)
        ret = re.sub(fr'({RE_V})s$', r'\1z', ret)
        return ret

    def get_Root(self):
        super().get_Root()
        if self.Auslautverhärtung:
            self.Root.value = self.cancel_final_devoicing(self.Root.value)


@dataclass
class _MjaStems(_MaStems):

    def derive_root_from_plural(self, plural):
        return re.sub(r'jos$', '', plural)

    @undefined
    def j_rules(self):
        pass

    # def get_Suffix(self):
    #     super().get_Suffix()
    #     for tag, suffix in self.Suffix.items():
    #         self.Suffix[tag] = self.j_rules('j' + suffix)  # pylint: disable=too-many-function-args


def sub(s, sub_rules):
    for before, after in sub_rules:
        s = re.sub(fr'{before}', fr'{after}', s)
    return s


@dataclass
class _NjaStems(_NaStems):

    def derive_root(self, s):
        return sub(s, [('i$', ''), ('au$', 'o')])

    def derive_root_from_plural(self, plural):
        return re.sub(r'ja$', '', plural)

    def j_rules(self, s):
        ret = re.sub(r'j$', 'i', s)
        ret = re.sub(r'js$', 'jis', ret)
        return ret

    # def get_Suffix(self):
    #     super().get_Suffix()
    #     for tag, suffix in self.Suffix.items():
    #         self.Suffix[tag] = self.j_rules('j' + suffix)


@reg_dc(5)
class Mja(_MjaStems):
    Lemma: str

    def derive_root(self, s):
        ret = re.sub(r'jis$', '', s)
        ret = re.sub(r'([ai])u$', r'\1w', ret)
        return ret

    def j_rules(self, s):
        ret = re.sub(r'j$', 'i', s)
        ret = re.sub(r'js$', 'jis', ret)
        return ret


@reg_dc(6)
class Mia(_MjaStems):

    def derive_root(self, s):
        return re.sub(r'eis$', '', s)

    def j_rules(self, s):
        sub_rules = [('j$', 'i'), ('ji?s$', 'eis')]
        return sub(s, sub_rules)


@reg_dc(7)
class Nja(_NjaStems):
    Lemma: str


@reg_dc(8)
class Nia(_NjaStems):
    Lemma: str


@reg_dc(9)
class Mwa(_MaStems):
    Lemma: str

    def derive_root(self, s):
        return sub(s, [('s$', ''), ('([ai])u$', '\1w')])


@reg_dc(10)
class Nwa(_NaStems):
    Lemma: str

    def derive_root(self, s):
        return sub(s, [('([ai])u$', '\1w')])


@dataclass
class _oStems(_VocStems):

    # suffix_lst = ["a", "a", "ai", "os", '', "os", "os", "om", "o", '']

    def derive_root(self, s):
        return sub(s, [('a$', '')])

    derive_root_from_plural = gen_sub([('os$', '')])


@reg_dc(11)
class Fo(_oStems):
    Lemma: str


@dataclass
class _joStems(_oStems):
    def derive_root(self, s):
        return sub(s, [('ja$', '')])

    derive_root_from_plural = gen_sub([('jos$', '')])

    # def get_Suffix(self):
    #     super().get_Suffix()
    #     for tag, suffix in self.Suffix.items():
    #         self.Suffix[tag] = 'j' + suffix


@reg_dc(12)
class Fjo(_joStems):
    Lemma: str

    def derive_root(self, s):
        return sub(s, [('ja$', '')])


@reg_dc(13)
class Fio(_joStems):
    Lemma: str

    def derive_root(self, s):
        return sub(s, [('i$', '')])

    # def get_Suffix(self):
    #     super().get_Suffix()
    #     self.Suffix['NOM-SING'] = 'i'


@reg_dc(14)
class Fwo(_oStems):
    Lemma: str


@dataclass
class _iStems(_VocStems):
    z_Assimilation: bool = False
    Auslautverhärtung: bool = True

    derive_root = gen_sub([('s$', ''), ('([ai])u$', '\1w')])
    derive_root_from_plural = gen_sub([('eis$', '')])
    cancel_final_devoicing = gen_sub([('({RE_V})f$', '\1b'),
                                      ('({RE_V})þ$', '\1d'),
                                      ('({RE_V})s$', '\1z')])

    # suffix_lst = {"s", "", '', '', '', "eis", "ins", "im", "e", ''}


@reg_dc(15)
class Mi(_iStems):
    Lemma: str
    # suffix_override = [('DAT-SING', 'a'), ('GEN-SING', 'is'), ('VOC-SING', None)]


@reg_dc(16)
class Fi(_iStems):
    Lemma: str

    # def get_Suffix(self):
    #     super().get_Suffix()
    #     self.Suffix['DAT-SING'] = 'ai'
    #     self.Suffix['GEN-SING'] = 'ais'


@reg_dc(17)
class FiO(Fi):
    Lemma: str

    derive_root_from_plural = gen_sub([('os$', '')])

    # def get_Suffix(self):
    #     super().get_Suffix()
    #     self.Suffix['NOM-PLUR'] = 'os'
    #     self.Suffix['GEN-PLUR'] = 'o'


@dataclass
class _uStems(_VocStems):
    derive_root = gen_sub([('us$', '')])
    derive_root_from_plural = gen_sub([('jus$', '')])

    # suffix_lst = ["us", "u", "au", "aus", 'au|u', "jus", "uns", "um", "iwe", '']

    # def get_Form(self):
    #     super().get_Form()
    #     if re.search(r'jus$', self.Lemma):
    #         self.Form.update_key('NOM-PLUR', None)
    #     if self.Pluraletantum:
    #         self.Form.update_key('NOM-PLUR', self.Lemma)


@reg_dc(18)
class U(_uStems):
    Lemma: str

    # form_remove = ['VOC-SING', 'PLUR']


@reg_dc(19)
class Mu(_uStems):
    Lemma: str


@reg_dc(20)
class Fu(_uStems):
    Lemma: str

    # form_remove = ['VOC']


@reg_dc(21)
class Nu(_uStems):

    derive_root = gen_sub([('u$', '')])

    @undefined
    def derive_root_from_plural(self):
        pass

    # suffix_override = [('NOM-SING', 'u'), ('VOC-SING', None), ('PLUR', None)]


@reg_dc(22)
class MuI(_uStems):
    derive_root_from_plural = gen_sub([('eis$', '')])
    # suffix_override = [('NOM-PLUR', 'eis'), ('GEN-PLUR', 'e')]


@dataclass
class _ConsStems(Noun):
    @undefined
    def phonology(self):
        pass


@reg_dc(23)
class Mn(_ConsStems):
    derive_root = gen_sub([('a$', '')])
    derive_root_from_plural = gen_sub([('ans$', '')])
    # suffix_lst = ["a", "an", "in", "ins", '', "ans", "ans", "am", "ane", '']


@reg_dc(24)
class Fn(_ConsStems):
    derive_root = gen_sub([('(o|ei)$', '')])
    derive_root_from_plural = gen_sub([('(o|ei)ns$', '')])

    # def get_Suffix(self):
    #     if re.search(r'o(ns)?$', self.Lemma):
    #         self.lemma_specific_suffix_lst = ["o", "on", "on", "ons", '', "ons", "ons", "om", "ono", '']
    #     if re.search(r'ei(ns)?$', self.Lemma):
    #         self.lemma_specific_suffix_lst = ["ei", "ein", "ein", "eins", '', "eins", "eins", "eim", "eino", '']
    #     super().get_Suffix()


@reg_dc(25)
class Nn(_ConsStems):
    derive_root = gen_sub([('o$', '')])
    derive_root_from_plural = gen_sub([('ona$', '')])
    # suffix_lst = ["o", "o", "in", "ins", '', "ona", "ona", "am", "ane", '']


@dataclass
class _rStems(_ConsStems):
    derive_root = gen_sub([('ar$', '')])
    derive_root_from_plural = gen_sub([('rjus$', '')])
    # suffix_lst = ["ar", "ar", "r", "rs", "ar", "rjus", "runs", "rum", "re", '']


@reg_dc(26)
class Mr(_rStems):
    pass


@reg_dc(27)
class Fr(_rStems):
    pass


@reg_dc(28)
class Mnd(_ConsStems):
    derive_root = gen_sub([('s$', '')])
    derive_root_from_plural = gen_sub([('s$', '')])
    # suffix_lst = ["s", "", "", "is", "", "s", "s", "am", "e", '']


@dataclass
class _RootNouns(Noun):
    Auslautverhärtung: bool = True
    derive_root = gen_sub([('s$', '')])
    derive_root_from_plural = gen_sub([('s$', '')])
    cancel_final_devoicing = gen_sub([('({RE_V})f$', '\1b'),
                                      ('({RE_V})þ$', '\1d'),
                                      ('({RE_V})s$', '\1z')])
    # suffix_lst = ["s", "", "", "s", '', "s", "s", '', "e", '']

    def get_Root(self):
        super().get_Root()
        if self.Auslautverhärtung:
            self.Root.value = self.cancel_final_devoicing(self.Root.value)


@reg_dc(29)
class Fkons(_RootNouns):
    # suffix_override = [('DAT-PLUR', 'im')]
    pass


@reg_dc(30)
class Mkons(_RootNouns):
    # suffix_override = [('DAT-PLUR', 'um')]
    pass


# TODO(j_luo) reg this
@dataclass
class Adjective(BaseMorphClass):
    Lemma: str
    WeakDeclensionOnly: bool = False
    StrongDeclensionOnly: bool = False

    @undefined
    def derive_root(self):
        pass

    @undefined
    def derive_root_from_weak_declension(self):
        pass

    phonology = gen_sub([('{RE_V})b(s?)$', '\1f\2'),
                         ('{RE_V})d(s?)$', '\1þ\2'),
                         ('({RE_V})z$', '\1s'),
                         ('o(?=e?i)', 'au'),
                         ('({RE_C}[ai])w(j|s?$)', '\1u\2')])

    reconstruction = gen_sub([('^(.*)$', r'\[\1\]')])

    # def get_Form(self):
    #     super().get_Form()
    #     if self.WeakDeclensionOnly:
    #         self.Form.update_key('STRONG', None)
    #     if self.StrongDeclensionOnly:
    #         self.Form.update('WEAK', None)

    def get_Root(self):
        super().get_Root()
        self.Root.value = self.derive_root_from_weak_declension(self.Lemma)  # pylint: disable=too-many-function-args


@dataclass
class _aStemsAdj(Adjective):
    derive_root = gen_sub([('s$', ''), ('([ai])u$', '\1w')])
    derive_root_from_weak_declension = gen_sub([('a$', '')])


@reg_dc(32)
class AdjA(_aStemsAdj):
    Lemma: str
    z_Assimilation: bool = False
    Auslautverhärtung: bool = True

    cancel_final_devoicing = gen_sub([('({RE_V})f$', '\1b'),
                                      ('({RE_V})þ$', '\1d'),
                                      ('({RE_V})s$', '\1z')])

    def get_Root(self):
        super().get_Root()
        if self.z_Assimilation:
            self.Root.value = self.Lemma
        if self.Auslautverhärtung:

            self.Root.value = self.cancel_final_devoicing(self.Root.value)


@dataclass
class _jaStemsAdj(_aStemsAdj):
    derive_root_from_weak_declension = gen_sub([('ja$', '')])

    @undefined
    def j_rules(self):
        pass


@reg_dc(33)
class AdjJa(_jaStemsAdj):
    derive_root = gen_sub([('jis$', ''), ('([ai])u$', '\1w')])
    j_rules = gen_sub([('j$', 'i'), ('js$', 'jis')])


@reg_dc(34)
class AdjIa(_jaStemsAdj):
    derive_root = gen_sub([('eis$', '')])
    j_rules = gen_sub([('j$', 'i'), ('ji?s$', 'eis')])


@reg_dc(35)
class AdjI(_jaStemsAdj):
    Lemma: str
    Auslautverhärtung: bool = True

    cancel_final_devoicing = gen_sub([('({RE_V})f$', '\1b'),
                                      ('({RE_V})þ$', '\1d'),
                                      ('({RE_V})s$', '\1z')])

    def get_Root(self):
        super().get_Root()
        if self.Auslautverhärtung:
            self.Root.value = self.cancel_final_devoicing(self.Root.value)


@reg_dc(36)
class AdjU(_jaStemsAdj):
    derive_root = gen_sub([('us$', '')])


@dataclass
class _AdjWithWeakFemEIN(_aStemsAdj):

    WeakDeclensionOnly: bool = False
    StrongDeclensionOnly: bool = False


@reg_dc(37)
class PartPres(_AdjWithWeakFemEIN):
    pass


@reg_dc(38)
class Comparative(_AdjWithWeakFemEIN):
    pass


@reg_dc(39)
class PartPerf(AdjA):
    pass


@reg_dc(40)
class Superlative(AdjA):
    pass

# TODO(j_luo)  reg this
@dataclass
class Verb(BaseMorphClass):
    Lemma: str

    @undefined
    def derive_root(self):
        pass


@reg_dc(42)
class _StrongVerb(Verb):
    derive_root = gen_sub([('an$', ''), ('^.*-', '')])

    @undefined
    def Ablaut2(self):
        pass

    @undefined
    def Ablaut3(self):
        pass

    @undefined
    def Ablaut4(self):
        pass

    def get_Root(self):
        base_root = self.derive_root(self.Lemma)
        self.Root['BASE'] = base_root
        self.Root['PRET'] = self.Ablaut3(base_root)
        self.Root['ACT-IND-PRET-SING'] = self.Ablaut2(base_root)
        self.Root['PART-PERF'] = self.Ablaut4(base_root)


@reg_dc(43)
class AblV1(_StrongVerb):
    Ablaut2 = gen_sub([('ei{RE__C}', 'ai')])
    Ablaut3 = gen_sub([('ei{RE__C}', 'i'), ('i(?={RE_C_BR}{RE_C}*$)', 'ai')])
    Ablaut4 = gen_sub([('ei{RE__C}', 'i'), ('i(?={RE_C_BR}{RE_C}*$)', 'ai')])


@reg_dc(44)
class AblV2(_StrongVerb):
    Ablaut2 = gen_sub([('i?u{RE__C}', 'au')])
    Ablaut3 = gen_sub([('i?u{RE__C}', 'u'), ('u(?={RE_C_BR}{RE_C}*$)', 'au')])
    Ablaut4 = gen_sub([('i?u{RE__C}', 'u'), ('u(?={RE_C_BR}{RE_C}*$)', 'au')])


@reg_dc(45)
class AblV3(_StrongVerb):
    Ablaut2 = gen_sub([('a?i{RE__C}', 'a')])
    Ablaut3 = gen_sub([('a?i{RE__C}', 'u'), ('u(?={RE_C_BR}{RE_C}*$)', 'au')])
    Ablaut4 = gen_sub([('a?i{RE__C}', 'u'), ('u(?={RE_C_BR}{RE_C}*$)', 'au')])


@reg_dc(46)
class AblV4(_StrongVerb):
    Ablaut2 = gen_sub([('a?i{RE__C}', 'a')])
    Ablaut3 = gen_sub([('a?i{RE__C}', 'e')])
    Ablaut4 = gen_sub([('a?i{RE__C}', 'u'), ('u(?={RE_C_BR}{RE_C}*$)', 'au')])


@reg_dc(47)
class AblV5(_StrongVerb):
    Ablaut2 = gen_sub([('a?i{RE__C}', 'a')])
    Ablaut3 = gen_sub([('a?i{RE__C}', 'e')])

    @undefined
    def Ablaut4(self):
        pass

# TODO(j_luo) reg this


class AblV6(_StrongVerb):
    Ablaut2 = gen_sub([('a?i{RE__C}', 'a')])
    Ablaut3 = gen_sub([('a?i{RE__C}', 'e')])

    @undefined
    def Ablaut4(self):
        pass


@dataclass
class _RedupVerb(_StrongVerb):

    @undefined
    def Ablaut2(self):
        pass

    @undefined
    def Ablaut3(self):
        pass

    @undefined
    def Ablaut4(self):
        pass


@reg_dc(48)
class RedV1(_RedupVerb):
    pass


@reg_dc(49)
class RedV2(_RedupVerb):
    pass


@reg_dc(50)
class RedV3(_RedupVerb):
    pass


@reg_dc(51)
class RedV4(_RedupVerb):
    pass


@reg_dc(52)
class RedV5(_RedupVerb):
    pass


@reg_dc(53)
class RedAblV(_RedupVerb):
    Ablaut2 = gen_sub([('(e(?={RE_C}$)|ai$)', 'o')])
    Ablaut3 = gen_sub([('(e(?={RE_C}$)|ai$)', 'o')])


@dataclass
class _WeakVerb(Verb):
    PretSuffix: ClassVar[str] = None

    def get_Root(self):
        self.Root['BASE'] = self.derive_root(self.Lemma)
        self.Root['PRET'] = self.Root['BASE'] + self.PretSuffix


@reg_dc(54)
class SWV1J(_WeakVerb):
    PretSuffix = 'id'
    derive_root = gen_sub([('jan$', ''), ('^.*-', ''), ('([ai])u$', '\1w')])


@reg_dc(55)
class SWV1I(SWV1J):
    pass


@reg_dc(56)
class SWV2(_WeakVerb):
    PretSuffix = 'od'
    derive_root = gen_sub([('on$', ''), ('^.*-', '')])


@reg_dc(57)
class SWV3(_WeakVerb):
    PretSuffix = 'aid'
    derive_root = gen_sub([('an$', ''), ('^.*-', '')])


@reg_dc(58)
class SWV4(_WeakVerb):
    PretSuffix = 'od'
    derive_root = gen_sub([('an$', ''), ('^.*-', '')])
