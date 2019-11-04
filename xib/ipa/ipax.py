from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Callable, List, Type

import inflection
import numpy as np

import xib.ipa

from . import ipa

# -------------------------------------------------------------- #
#                  Useful functions and classes                  #
# -------------------------------------------------------------- #


conversions = dict()
reverse_conversions = dict()
_registered = dict()


def convert(enum: Enum) -> Callable[[Enum], Enum]:

    def wrapper(cls):
        old_names = set([x.name for x in enum])
        new_names = set([x.name for x in cls])
        if old_names != new_names:
            raise RuntimeError(f'Both enums should have the same constants, but got {old_names} and {new_names}.')

        # Register all enum classes that are converted.
        _registered[cls.__name__] = cls

        for feat in enum:
            if feat in conversions:
                raise RuntimeError(f'The same index is converted twice.')
            new_feat = cls[feat.name]
            if new_feat in reverse_conversions:
                raise RuntimeError(f'The same feature is converted twice.')
            conversions[feat] = new_feat
            reverse_conversions[new_feat] = feat

        return cls

    return wrapper


class BaseEnumValue(ABC):

    @abstractmethod
    def __sub__(self, other):
        pass

    @classmethod
    @abstractmethod
    def is_complex(cls) -> bool:
        pass


@dataclass
class AutoIndex(BaseEnumValue):
    total_cat = -1
    total_idx = -1

    g_idx: int
    c_idx: int
    f_idx: int

    def __sub__(self, other: 'AutoIndex') -> float:
        if self.c_idx != other.c_idx:
            raise TypeError(f'Should not call __sub__ for indices that are in different categories.')
        diff_g = self.g_idx - other.g_idx
        diff_f = self.f_idx - other.f_idx
        if diff_g != diff_f:
            raise RuntimeError('Something is seriously wrong.')
        return diff_g

    @classmethod
    def is_complex(cls) -> bool:
        return False


@dataclass
class Factors(BaseEnumValue):
    """Base class for CPP and CMP that autuomatically converts a string attribute to an enum element."""

    def __len__(self):
        return len(self.__dataclass_fields__)

    @classmethod
    def is_complex(cls) -> bool:
        return True

    def __post_init__(self):
        for name, field in self.__dataclass_fields__.items():
            cls = field.type
            old_value = getattr(self, name)
            new_value = cls[old_value]
            setattr(self, name, new_value)

    def __iter__(self):
        for name in self.__dataclass_fields__:
            yield getattr(self, name)

    def __sub__(self, other: 'Factors') -> float:
        if type(self) is not type(other):
            raise TypeError(f'Mismatched types, got {type(self)} and {type(other)}.')

        ret = 0.0
        for name, field in self.__dataclass_fields__.items():
            cls = field.type

            self_value = getattr(self, name)
            other_value = getattr(other, name)

            ret += cls.get_distance(self_value, other_value)
        return ret


class DistEnum(Enum):
    # TODO(j_luo) really need a good way to doing enum. stdlib is not very flexible.
    """An Enum class that has defined distance functions."""

    @classmethod
    def get_distance_matrix(cls) -> np.ndarray:
        if hasattr(cls, '_cached_distance_matrix'):
            return cls._cached_distance_matrix

        n = len(cls)
        ret = np.zeros([n, n], dtype='float32')
        for i, feat_i in enumerate(cls):
            for j, feat_j in enumerate(cls):
                ret[i, j] = cls.get_distance(feat_i, feat_j)
        inf_mask = ret == np.inf
        ret[inf_mask] = 0.0
        if ret.max() == 0:
            raise RuntimeError('Something is terribly wrong.')
        ret = ret / ret.max()
        ret[inf_mask] = 1.0

        cls._cached_distance_matrix = ret
        return ret

    @classmethod
    def get_distance(cls, feat1, feat2):
        ret = cls._get_distance(feat1, feat2)
        if not isinstance(ret, (float, int)):
            raise TypeError(f'Wrong type, got {type(ret)}.')
        if ret < 0.0:
            raise ValueError(f'Got negative value {ret}.')
        return ret

    @classmethod
    def _get_distance(cls, feat1, feat2):
        raise NotImplementedError()

    def _generate_next_value_(name, start, count, last_values):
        if not last_values:
            AutoIndex.total_cat += 1
            f_idx = 0
        else:
            f_idx = last_values[-1].f_idx + 1
        AutoIndex.total_idx += 1
        return AutoIndex(AutoIndex.total_idx, AutoIndex.total_cat, f_idx)

    @classmethod
    def num_groups(cls):
        raise NotImplementedError()

    @classmethod
    def get_name(cls):
        return xib.ipa.Name(cls.__name__, 'camel')


def _is_special(feat):
    """Whether `feat` is special: 'NONE' or some discrete value marked by 'DISC_' prefix."""
    if feat.name == 'NONE':
        return True
    return feat.name.startswith('DISC_')


class ContinuousEnum(DistEnum):

    @classmethod
    def _get_distance(cls, feat1, feat2):
        if feat1.name == feat2.name and _is_special(feat1):
            return 0.0
        if _is_special(feat1) or _is_special(feat2):
            return np.inf
        return abs(feat1.value - feat2.value)


class DiscreteEnum(DistEnum):

    @classmethod
    def _get_distance(cls, feat1, feat2):
        if feat1.name == feat2.name == 'NONE':
            return 0.0
        if feat1.name == 'NONE' or feat2.name == 'NONE':
            return np.inf
        return 0.0 if feat1 is feat2 else 1.0


# -------------------------------------------------------------- #
#                            Main body                           #
# -------------------------------------------------------------- #

@unique
class CategoryX(Enum):
    PTYPE_X = 0
    C_VOICING_X = 1
    C_PLACE_X = 2
    C_MANNER_X = 3
    V_HEIGHT_X = 4
    V_BACKNESS_X = 5
    V_ROUNDNESS_X = 6

    @classmethod
    def get_enum(cls, name: str) -> Type[DistEnum]:
        return _registered[inflection.camelize(name.lower())]


@unique
@convert(ipa.Ptype)
class PtypeX(DiscreteEnum):
    CONSONANT = auto()
    VOWEL = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
@convert(ipa.CVoicing)
class CVoicingX(DiscreteEnum):
    NONE = auto()
    VOICED = auto()
    VOICELESS = auto()

    @classmethod
    def num_groups(cls):
        return 1


# Based on https://en.wikipedia.org/wiki/Place_of_articulation#Table_of_gestures_and_passive_articulators_and_resulting_places_of_articulation.
@unique
class CPlaceActiveArticulator(ContinuousEnum):
    NONE = auto()
    LABIAL = auto()
    # Coronal contains three subcategories: laminal, apical and subapical.
    # The first two are marked in Diacritics while the third one doesn't present any contrast in the paradigm used by ipapy.
    CORONAL = auto()
    DORSAL = auto()
    RADICAL = auto()
    LARYNGEAL = auto()

    DISC_LABIO_ALVEOLAR = auto()
    DISC_LABIO_PALATAL = auto()
    DISC_LABIO_VELAR = auto()
    DISC_PALATO_ALVEOLO_VELAR = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
class CPlacePassiveArticulator(ContinuousEnum):
    NONE = auto()
    UPPER_LIP = auto()
    # Upper teeth contains two subcategories -- upper teeth and upper teeth/alveolar ridge -- which don't have any contrast in this paradigm.
    UPPER_TEETH = auto()
    ALVEOLAR_RIDGE = auto()
    POSTALVEOLAR = auto()
    HARD_PALATE = auto()
    SOFT_PALATE = auto()
    UVULA = auto()
    PHARYNX = auto()
    EPIGLOTTIS = auto()
    GLOTTIS = auto()

    DISC_LABIO_ALVEOLAR = auto()
    DISC_LABIO_PALATAL = auto()
    DISC_LABIO_VELAR = auto()
    DISC_PALATO_ALVEOLO_VELAR = auto()

    @classmethod
    def num_groups(cls):
        return 1


@dataclass
class CPP(Factors):  # Stands for consonant place parameter.
    active: CPlaceActiveArticulator
    passive: CPlacePassiveArticulator


@unique
@convert(ipa.CPlace)
class CPlaceX(ContinuousEnum):
    NONE = CPP('NONE', 'NONE')
    ALVEOLAR = CPP('CORONAL', 'ALVEOLAR_RIDGE')
    ALVEOLO_PALATAL = CPP('DORSAL', 'POSTALVEOLAR')
    BILABIAL = CPP('LABIAL', 'UPPER_LIP')
    DENTAL = CPP('CORONAL', 'UPPER_TEETH')
    GLOTTAL = CPP('LARYNGEAL', 'GLOTTIS')
    LABIO_DENTAL = CPP('LABIAL', 'UPPER_TEETH')
    PALATAL = CPP('DORSAL', 'HARD_PALATE')
    PALATO_ALVEOLAR = CPP('CORONAL', 'POSTALVEOLAR')
    PHARYNGEAL = CPP('RADICAL', 'PHARYNX')
    RETROFLEX = CPP('CORONAL', 'HARD_PALATE')
    UVULAR = CPP('DORSAL', 'UVULA')
    VELAR = CPP('DORSAL', 'SOFT_PALATE')

    # Four places that are weird.
    # LABIO_ALVEOLAR, LABIO_PALATAL, LABIO_VELAR, PALATO_ALVEOLO_VELAR.
    LABIO_ALVEOLAR = CPP('DISC_LABIO_ALVEOLAR', 'DISC_LABIO_ALVEOLAR')
    LABIO_PALATAL = CPP('DISC_LABIO_PALATAL', 'DISC_LABIO_PALATAL')
    LABIO_VELAR = CPP('DISC_LABIO_VELAR', 'DISC_LABIO_VELAR')
    PALATO_ALVEOLO_VELAR = CPP('DISC_PALATO_ALVEOLO_VELAR', 'DISC_PALATO_ALVEOLO_VELAR')

    @classmethod
    def num_groups(cls):
        return 2

    @classmethod
    def parts(cls) -> List[DistEnum]:
        return [field.type for field in CPP.__dataclass_fields__.values()]


@unique
class CMannerSonority(ContinuousEnum):
    NONE = auto()
    STOP = auto()
    AFFRICATE = auto()
    FRICATIVE = auto()
    NASAL = auto()
    LIQUID = auto()
    APPROXIMANT = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
class CMannerNasality(DiscreteEnum):
    NONE = auto()
    NASAL = auto()
    NON_NASAL = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
class CMannerLaterality(DiscreteEnum):
    # NOTE(j_luo) Only mark +lat or -lat then there are two contrastive categories.
    NONE = auto()
    LATERAL = auto()
    NON_LATERAL = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
class CMannerAirstream(DiscreteEnum):
    NONE = auto()
    PULMONIC_EGRESSIVE = auto()
    GLOTTALIC_EGRESSIVE = auto()
    GLOTTALIC_INGRESSIVE = auto()
    LINGUAL_INGRESSIVE = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
class CMannerSibilance(DiscreteEnum):
    NONE = auto()
    SIBILANT = auto()
    NON_SIBILANT = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
class CMannerVibrancy(DiscreteEnum):
    NONE = auto()
    FLAP = auto()
    TRILL = auto()

    @classmethod
    def num_groups(cls):
        return 1


@dataclass
class CMP(Factors):  # Stands for consonant manner parameter.
    sonority: CMannerSonority
    nasality: CMannerNasality
    laterality: CMannerLaterality
    airstream: CMannerAirstream
    sibilance: CMannerSibilance
    vibrancy: CMannerVibrancy


@unique
@convert(ipa.CManner)
class CMannerX(ContinuousEnum):
    NONE = CMP('NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE')
    APPROXIMANT = CMP('APPROXIMANT', 'NON_NASAL', 'NON_LATERAL', 'PULMONIC_EGRESSIVE', 'NONE', 'NONE')
    CLICK = CMP('NONE', 'NON_NASAL', 'NON_LATERAL', 'LINGUAL_INGRESSIVE', 'NONE', 'NONE')
    # TODO(j_luo) There might be some bug here at https://github.com/pettarin/ipapy/blob/4026e4e27e857820c12895a7d2e7281b1cff7723/ipapy/data/kirshenbaum.dat#L453,
    # where an ejective fricative is misclassified as an ejective stop.
    EJECTIVE = CMP('STOP', 'NON_NASAL', 'NONE', 'GLOTTALIC_EGRESSIVE', 'NONE', 'NONE')
    EJECTIVE_AFFRICATE = CMP('AFFRICATE', 'NON_NASAL', 'NON_LATERAL', 'GLOTTALIC_EGRESSIVE', 'NONE', 'NONE')
    EJECTIVE_FRICATIVE = CMP('FRICATIVE', 'NON_NASAL', 'NONE', 'GLOTTALIC_EGRESSIVE', 'NONE', 'NONE')
    FLAP = CMP('LIQUID', 'NON_NASAL', 'NON_LATERAL', 'PULMONIC_EGRESSIVE', 'NONE', 'FLAP')
    # Set to STOP. See https://en.wikipedia.org/wiki/Implosive_consonant#Types.
    IMPLOSIVE = CMP('STOP', 'NON_NASAL', 'NONE', 'GLOTTALIC_INGRESSIVE', 'NONE', 'NONE')
    LATERAL_AFFRICATE = CMP('AFFRICATE', 'NON_NASAL', 'LATERAL', 'PULMONIC_EGRESSIVE', 'NONE', 'NONE')
    # NOTE(j_luo) This is classified as a liquid, instead of a normal approximant.
    LATERAL_APPROXIMANT = CMP('LIQUID', 'NON_NASAL', 'LATERAL', 'PULMONIC_EGRESSIVE', 'NONE', 'NONE')
    LATERAL_CLICK = CMP('NONE', 'NON_NASAL', 'LATERAL', 'LINGUAL_INGRESSIVE', 'NONE', 'NONE')
    LATERAL_EJECTIVE_AFFRICATE = CMP('AFFRICATE', 'NON_NASAL', 'LATERAL', 'GLOTTALIC_EGRESSIVE', 'NONE', 'NONE')
    LATERAL_FLAP = CMP('LIQUID', 'NON_NASAL', 'LATERAL', 'PULMONIC_EGRESSIVE', 'NONE', 'FLAP')
    LATERAL_FRICATIVE = CMP('FRICATIVE', 'NON_NASAL', 'LATERAL', 'PULMONIC_EGRESSIVE', 'NONE', 'NONE')
    NASAL = CMP('NASAL', 'NASAL', 'NONE', 'PULMONIC_EGRESSIVE', 'NONE', 'NONE')
    NON_SIBILANT_AFFRICATE = CMP('AFFRICATE', 'NON_NASAL', 'NONE', 'PULMONIC_EGRESSIVE', 'NON_SIBILANT', 'NONE')
    NON_SIBILANT_FRICATIVE = CMP('FRICATIVE', 'NON_NASAL', 'NONE', 'PULMONIC_EGRESSIVE', 'NON_SIBILANT', 'NONE')
    PLOSIVE = CMP('STOP', 'NON_NASAL', 'NONE', 'PULMONIC_EGRESSIVE', 'NONE', 'NONE')
    SIBILANT_AFFRICATE = CMP('AFFRICATE', 'NON_NASAL', 'NONE', 'PULMONIC_EGRESSIVE', 'SIBILANT', 'NONE')
    SIBILANT_FRICATIVE = CMP('FRICATIVE', 'NON_NASAL', 'NONE', 'PULMONIC_EGRESSIVE', 'SIBILANT', 'NONE')
    TRILL = CMP('LIQUID', 'NON_NASAL', 'NONE', 'PULMONIC_EGRESSIVE', 'NONE', 'TRILL')

    @classmethod
    def num_groups(cls):
        return 6

    @classmethod
    def parts(cls) -> List[DistEnum]:
        return [field.type for field in CMP.__dataclass_fields__.values()]


@unique
@convert(ipa.VHeight)
class VHeightX(ContinuousEnum):
    NONE = auto()
    CLOSE = auto()
    NEAR_CLOSE = auto()
    CLOSE_MID = auto()
    MID = auto()
    OPEN_MID = auto()
    NEAR_OPEN = auto()
    OPEN = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
@convert(ipa.VBackness)
class VBacknessX(ContinuousEnum):
    NONE = auto()
    BACK = auto()
    NEAR_BACK = auto()
    CENTRAL = auto()
    NEAR_FRONT = auto()
    FRONT = auto()

    @classmethod
    def num_groups(cls):
        return 1


@unique
@convert(ipa.VRoundness)
class VRoundnessX(DiscreteEnum):
    NONE = auto()
    ROUNDED = auto()
    UNROUNDED = auto()

    @classmethod
    def num_groups(cls):
        return 1
