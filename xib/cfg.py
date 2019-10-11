from dataclasses import dataclass
from enum import Enum, unique

import inflection

from arglib import Registry

reg = Registry('cfg')


@dataclass
class Index:
    g_idx: int  # global index
    c_idx: int  # category index
    f_idx: int  # feature index

    _instances = dict()

    def __post_init__(self):
        if self.g_idx in self._instances:
            raise RuntimeError(f'Duplicate global index at {self.g_idx}.')
        self._instances[self.g_idx] = self

    @classmethod
    def get_feature(cls, g_idx):
        index = cls._instances[g_idx]
        cat = Category(index.c_idx)
        cat_cls = _get_cat_cls_by_enum(cat)
        return cat_cls(index)

    @classmethod
    def total_indices(cls):
        return len(cls._instances)


def _get_cat_cls_by_enum(cat):
    return globals()[inflection.camelize(cat.name.lower())]


def _get_enum_by_cat_cls(cat_cls):
    return getattr(Category, inflection.underscore(cat_cls.__name__).upper())


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
    def get_named_cat_enums(cls):
        for x in cls:
            cat_cls = _get_cat_cls_by_enum(x)
            yield x.name, cat_cls


@unique
class Ptype(Enum):
    CONSONANT = Index(0, 0, 0)
    VOWEL = Index(1, 0, 1)


conditions = dict()


def _conditioned_on_cls(e):  # e stands for enum.

    def decorator(cls):
        conditions[_get_enum_by_cat_cls(cls).name] = e.value
        return cls

    return decorator


@unique
@_conditioned_on_cls(Ptype.CONSONANT)
class CVoicing(Enum):
    NONE = Index(2, 1, 0)
    VOICED = Index(3, 1, 1)
    VOICELESS = Index(4, 1, 2)


@unique
@_conditioned_on_cls(Ptype.CONSONANT)
class CPlace(Enum):
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
@_conditioned_on_cls(Ptype.CONSONANT)
class CManner(Enum):
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
@_conditioned_on_cls(Ptype.VOWEL)
class VHeight(Enum):
    NONE = Index(43, 4, 0)
    CLOSE = Index(44, 4, 1)
    CLOSE_MID = Index(45, 4, 2)
    MID = Index(46, 4, 3)
    NEAR_CLOSE = Index(47, 4, 4)
    NEAR_OPEN = Index(48, 4, 5)
    OPEN = Index(49, 4, 6)
    OPEN_MID = Index(50, 4, 7)


@unique
@_conditioned_on_cls(Ptype.VOWEL)
class VBackness(Enum):
    NONE = Index(51, 5, 0)
    BACK = Index(52, 5, 1)
    CENTRAL = Index(53, 5, 2)
    FRONT = Index(54, 5, 3)
    NEAR_BACK = Index(55, 5, 4)
    NEAR_FRONT = Index(56, 5, 5)


@unique
@_conditioned_on_cls(Ptype.VOWEL)
class VRoundness(Enum):
    NONE = Index(57, 6, 0)
    ROUNDED = Index(58, 6, 1)
    UNROUNDED = Index(59, 6, 2)


@unique
class Diacritics(Enum):
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
class SStress(Enum):
    NONE = Index(87, 8, 0)
    PRIMARY_STRESS = Index(88, 8, 1)


@unique
class SLength(Enum):
    NONE = Index(89, 9, 0)
    EXTRA_SHORT = Index(90, 9, 1)
    HALF_LONG = Index(91, 9, 2)
    LONG = Index(92, 9, 3)


@unique
class SBreak(Enum):
    NONE = Index(93, 10, 0)
    LINKING = Index(94, 10, 1)
    WORD_BREAK = Index(95, 10, 2)


@unique
class TLevel(Enum):
    NONE = Index(96, 11, 0)
    EXTRA_HIGH_LEVEL = Index(97, 11, 1)
    EXTRA_LOW_LEVEL = Index(98, 11, 2)
    HIGH_LEVEL = Index(99, 11, 3)
    LOW_LEVEL = Index(100, 11, 4)
    MID_LEVEL = Index(101, 11, 5)


@unique
class TContour(Enum):
    NONE = Index(102, 12, 0)
    FALLING_CONTOUR = Index(103, 12, 1)
    HIGH_MID_FALLING_CONTOUR = Index(104, 12, 2)
    HIGH_RISING_CONTOUR = Index(105, 12, 3)
    LOW_RISING_CONTOUR = Index(106, 12, 4)
    MID_LOW_FALLING_CONTOUR = Index(107, 12, 5)
    RISING_CONTOUR = Index(108, 12, 6)
    RISING_FALLING_CONTOUR = Index(109, 12, 7)


@unique
class TGlobal(Enum):
    NONE = Index(110, 13, 0)
    DOWNSTEP = Index(111, 13, 1)


@reg
class TestEn:
    data_path: str = 'data/phones_en_idx.pth'
    dim: int = 20
    num_features: int = 112
    num_feature_groups: int = 14
    check_interval: int = 50
    char_per_batch: int = 2000
    num_steps: int = 10000
    window_size: int = 3


@reg
class TestEnP(TestEn):  # Only deal with ptype.
    mode: str = 'p'


@reg
class TestEnPCV(TestEn):
    mode: str = 'pcv'


@reg
class TestEnPDST(TestEn):
    mode: str = 'pdst'
