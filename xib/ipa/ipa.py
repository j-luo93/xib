"""
Just to clarify some terminologies and variable names.
Category: the main enum class.
category or cat: the class variable in Category class. Category.PTYPE is one example.
enum or e: the Enum class corresponding to one feature group. PType for instance.
IPAFeature: the base Enum class that all feature Enum classes subclass.
feature or feat: the IPAFeature variable. PType.CONSONANT for instance.
"""
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, unique
from typing import Type

import inflection

import xib.ipa


def get_enum_by_cat(cat) -> Type['IPAFeature']:
    return globals()[inflection.camelize(cat.name.lower())]


def get_cat_by_enum(e):
    return getattr(Category, inflection.underscore(e.__name__).upper())


@unique
class Category(Enum):
    PTYPE = 0
    C_VOICING = 1
    C_PLACE = 2
    C_MANNER = 3
    V_HEIGHT = 4
    V_BACKNESS = 5
    V_ROUNDNESS = 6
    DIACRITICS = 7
    S_STRESS = 8
    S_LENGTH = 9
    S_BREAK = 10
    T_LEVEL = 11
    T_CONTOUR = 12
    T_GLOBAL = 13

    @classmethod
    def get_enum(cls, name: str) -> Type['IPAFeature']:
        return globals()[inflection.camelize(name.lower())]


class IPAFeature(Enum):

    @classmethod
    def get(cls, f_idx: int) -> str:
        """Get the feature name from feature index."""
        cat = get_cat_by_enum(cls)
        c_idx = cat.value
        return Index.get_feature_by_cat(c_idx, f_idx).name

    @classmethod
    def get_name(cls):
        return xib.ipa.Name(cls.__name__, 'camel')


@dataclass(frozen=True)
class Index:
    g_idx: int  # global index
    c_idx: int  # category index
    f_idx: int  # feature index

    _instances = dict()
    _cat_instances = defaultdict(dict)

    def __post_init__(self):
        if self.g_idx in self._instances:
            raise RuntimeError(f'Duplicate global index at {self.g_idx}.')
        self._instances[self.g_idx] = self
        if self.f_idx in self._cat_instances[self.c_idx]:
            raise RuntimeError(f'Duplicate feature index at {self.f_idx} for category index {self.c_idx}.')
        self._cat_instances[self.c_idx][self.f_idx] = self

    @classmethod
    def get_feature(cls, g_idx: int) -> IPAFeature:
        """Get an IPAFeature instance by the global index."""
        index = cls._instances[g_idx]
        return Index._get_feature_by_index(index)

    @staticmethod
    def _get_feature_by_index(index):
        cat = Category(index.c_idx)
        e = get_enum_by_cat(cat)
        return e(index)

    @classmethod
    def get_feature_by_cat(cls, c_idx: int, f_idx: int) -> IPAFeature:
        """Get an IPAFeature instance by the category index and the feature idx."""
        index = cls._cat_instances[c_idx][f_idx]
        return Index._get_feature_by_index(index)

    @classmethod
    def total_indices(cls):
        return len(cls._instances)


@unique
class Ptype(IPAFeature):
    CONSONANT = Index(0, 0, 0)
    VOWEL = Index(1, 0, 1)


conditions = dict()  # cat => index


def _conditioned_on_feat(feat):

    def decorator(e):
        conditions[get_cat_by_enum(e)] = feat.value
        return e

    return decorator


no_none_predictions = dict()  # cat => index


def _avoid_predicting_none(e):
    no_none_predictions[get_cat_by_enum(e)] = e.NONE.value
    return e


@unique
@_avoid_predicting_none
@_conditioned_on_feat(Ptype.CONSONANT)
class CVoicing(IPAFeature):
    NONE = Index(2, 1, 0)
    VOICED = Index(3, 1, 1)
    VOICELESS = Index(4, 1, 2)


@unique
@_avoid_predicting_none
@_conditioned_on_feat(Ptype.CONSONANT)
class CPlace(IPAFeature):
    NONE = Index(5, 2, 0)
    ALVEOLAR = Index(6, 2, 1)
    ALVEOLO_PALATAL = Index(7, 2, 2)
    BILABIAL = Index(8, 2, 3)
    DENTAL = Index(9, 2, 4)
    GLOTTAL = Index(10, 2, 5)
    LABIO_ALVEOLAR = Index(11, 2, 6)
    LABIO_DENTAL = Index(12, 2, 7)
    LABIO_PALATAL = Index(13, 2, 8)
    LABIO_VELAR = Index(14, 2, 9)
    PALATAL = Index(15, 2, 10)
    PALATO_ALVEOLAR = Index(16, 2, 11)
    PALATO_ALVEOLO_VELAR = Index(17, 2, 12)
    PHARYNGEAL = Index(18, 2, 13)
    RETROFLEX = Index(19, 2, 14)
    UVULAR = Index(20, 2, 15)
    VELAR = Index(21, 2, 16)


@unique
@_avoid_predicting_none
@_conditioned_on_feat(Ptype.CONSONANT)
class CManner(IPAFeature):
    NONE = Index(22, 3, 0)
    APPROXIMANT = Index(23, 3, 1)
    CLICK = Index(24, 3, 2)
    EJECTIVE = Index(25, 3, 3)
    EJECTIVE_AFFRICATE = Index(26, 3, 4)
    EJECTIVE_FRICATIVE = Index(27, 3, 5)
    FLAP = Index(28, 3, 6)
    IMPLOSIVE = Index(29, 3, 7)
    LATERAL_AFFRICATE = Index(30, 3, 8)
    LATERAL_APPROXIMANT = Index(31, 3, 9)
    LATERAL_CLICK = Index(32, 3, 10)
    LATERAL_EJECTIVE_AFFRICATE = Index(33, 3, 11)
    LATERAL_FLAP = Index(34, 3, 12)
    LATERAL_FRICATIVE = Index(35, 3, 13)
    NASAL = Index(36, 3, 14)
    NON_SIBILANT_AFFRICATE = Index(37, 3, 15)
    NON_SIBILANT_FRICATIVE = Index(38, 3, 16)
    PLOSIVE = Index(39, 3, 17)
    SIBILANT_AFFRICATE = Index(40, 3, 18)
    SIBILANT_FRICATIVE = Index(41, 3, 19)
    TRILL = Index(42, 3, 20)


@unique
@_avoid_predicting_none
@_conditioned_on_feat(Ptype.VOWEL)
class VHeight(IPAFeature):
    NONE = Index(43, 4, 0)
    CLOSE = Index(44, 4, 1)
    CLOSE_MID = Index(45, 4, 2)
    MID = Index(46, 4, 3)
    NEAR_CLOSE = Index(47, 4, 4)
    NEAR_OPEN = Index(48, 4, 5)
    OPEN = Index(49, 4, 6)
    OPEN_MID = Index(50, 4, 7)


@unique
@_avoid_predicting_none
@_conditioned_on_feat(Ptype.VOWEL)
class VBackness(IPAFeature):
    NONE = Index(51, 5, 0)
    BACK = Index(52, 5, 1)
    CENTRAL = Index(53, 5, 2)
    FRONT = Index(54, 5, 3)
    NEAR_BACK = Index(55, 5, 4)
    NEAR_FRONT = Index(56, 5, 5)


@unique
@_avoid_predicting_none
@_conditioned_on_feat(Ptype.VOWEL)
class VRoundness(IPAFeature):
    NONE = Index(57, 6, 0)
    ROUNDED = Index(58, 6, 1)
    UNROUNDED = Index(59, 6, 2)


@unique
class Diacritics(IPAFeature):
    NONE = Index(60, 7, 0)
    ADVANCED = Index(61, 7, 1)
    ADVANCED_TONGUE_ROOT = Index(62, 7, 2)
    APICAL = Index(63, 7, 3)
    ASPIRATED = Index(64, 7, 4)
    BREATHY_VOICED = Index(65, 7, 5)
    CENTRALIZED = Index(66, 7, 6)
    CREAKY_VOICED = Index(67, 7, 7)
    LABIALIZED = Index(68, 7, 8)
    LAMINAL = Index(69, 7, 9)
    LATERAL_RELEASE = Index(70, 7, 10)
    LESS_ROUNDED = Index(71, 7, 11)
    LOWERED = Index(72, 7, 12)
    MORE_ROUNDED = Index(73, 7, 13)
    NASALIZED = Index(74, 7, 14)
    NO_AUDIBLE_RELEASE = Index(75, 7, 15)
    NON_SYLLABIC = Index(76, 7, 16)
    PALATALIZED = Index(77, 7, 17)
    PHARYNGEALIZED = Index(78, 7, 18)
    RAISED = Index(79, 7, 19)
    RETRACTED = Index(80, 7, 20)
    RETRACTED_TONGUE_ROOT = Index(81, 7, 21)
    RHOTACIZED = Index(82, 7, 22)
    SYLLABIC = Index(83, 7, 23)
    TIE_BAR_ABOVE = Index(84, 7, 24)
    TIE_BAR_BELOW = Index(85, 7, 25)
    VELARIZED = Index(86, 7, 26)


@unique
class SStress(IPAFeature):
    NONE = Index(87, 8, 0)
    PRIMARY_STRESS = Index(88, 8, 1)


@unique
class SLength(IPAFeature):
    NONE = Index(89, 9, 0)
    EXTRA_SHORT = Index(90, 9, 1)
    HALF_LONG = Index(91, 9, 2)
    LONG = Index(92, 9, 3)


@unique
class SBreak(IPAFeature):
    NONE = Index(93, 10, 0)
    LINKING = Index(94, 10, 1)
    WORD_BREAK = Index(95, 10, 2)


@unique
class TLevel(IPAFeature):
    NONE = Index(96, 11, 0)
    EXTRA_HIGH_LEVEL = Index(97, 11, 1)
    EXTRA_LOW_LEVEL = Index(98, 11, 2)
    HIGH_LEVEL = Index(99, 11, 3)
    LOW_LEVEL = Index(100, 11, 4)
    MID_LEVEL = Index(101, 11, 5)


@unique
class TContour(IPAFeature):
    NONE = Index(102, 12, 0)
    FALLING_CONTOUR = Index(103, 12, 1)
    HIGH_MID_FALLING_CONTOUR = Index(104, 12, 2)
    HIGH_RISING_CONTOUR = Index(105, 12, 3)
    LOW_RISING_CONTOUR = Index(106, 12, 4)
    MID_LOW_FALLING_CONTOUR = Index(107, 12, 5)
    RISING_CONTOUR = Index(108, 12, 6)
    RISING_FALLING_CONTOUR = Index(109, 12, 7)


@unique
class TGlobal(IPAFeature):
    NONE = Index(110, 13, 0)
    DOWNSTEP = Index(111, 13, 1)
