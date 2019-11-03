from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Callable

import numpy as np

from .ipa import CManner, CPlace

# -------------------------------------------------------------- #
#                  Useful functions and classes                  #
# -------------------------------------------------------------- #


def convert(enum: Enum) -> Callable[[Enum], Enum]:
    # FIXME(j_luo)
    return enum


class DistEnum(Enum):
    """An Enum class that has defined distance functions."""

    @classmethod
    def get_distance_matrix(cls) -> np.ndarray:
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
        if last_values:
            return last_values[-1] + 1
        else:
            return start - 1


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
class PtypeX(DiscreteEnum):
    CONSONANT = auto()
    VOWEL = auto()


@unique
class CVoicingX(DiscreteEnum):
    NONE = auto()
    VOICED = auto()
    VOICELESS = auto()


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


@dataclass
class Factors:
    """Base class for CPP and CMP that autuomatically converts a string attribute to an enum element."""

    def __len__(self):
        return len(self.__dataclass_fields__)

    def __post_init__(self):
        for name, field in self.__dataclass_fields__.items():
            cls = field.type
            old_value = getattr(self, name)
            new_value = cls[old_value]
            setattr(self, name, new_value)

    def __sub__(self, other: 'Factors') -> float:
        if type(self) is not type(other):
            raise TypeError(f'Mismatched types, got {type(self)} and {type(other)}.')

        ret = 0.0
        for name, field in self.__dataclass_fields__.items():
            # The the maximum distance.
            cls = field.type

            self_value = getattr(self, name)
            other_value = getattr(other, name)

            if self_value.name == 'NONE' or other_value.name == 'NONE':
                raise RuntimeError(f'Should not use this operator on NONE values.')

            # Normalize the score -- some categories have more elements.
            ret += cls.get_distance(self_value, other_value)
        return ret


@dataclass
class CPP(Factors):  # Stands for consonant place parameter.
    active: CPlaceActiveArticulator
    passive: CPlacePassiveArticulator


# @convert(CPlace)
@unique
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


@unique
class CMannerSonority(ContinuousEnum):
    NONE = auto()
    STOP = auto()
    AFFRICATE = auto()
    FRICATIVE = auto()
    NASAL = auto()
    LIQUID = auto()
    APPROXIMANT = auto()


@unique
class CMannerNasality(DiscreteEnum):
    NONE = auto()
    NASAL = auto()
    NON_NASAL = auto()


@unique
class CMannerLaterality(DiscreteEnum):
    # NOTE(j_luo) Only mark +lat or -lat then there are two contrastive categories.
    NONE = auto()
    LATERAL = auto()
    NON_LATERAL = auto()


@unique
class CMannerAirstream(DiscreteEnum):
    NONE = auto()
    PULMONIC_EGRESSIVE = auto()
    GLOTTALIC_EGRESSIVE = auto()
    GLOTTALIC_INGRESSIVE = auto()
    LINGUAL_INGRESSIVE = auto()


@unique
class CMannerSibilance(DiscreteEnum):
    NONE = auto()
    SIBILANT = auto()
    NON_SIBILANT = auto()


@unique
class CMannerVibrancy(DiscreteEnum):
    NONE = auto()
    FLAP = auto()
    TRILL = auto()


@dataclass
class CMP(Factors):  # Stands for consonant manner parameter.
    sonority: CMannerSonority
    nasality: CMannerNasality
    laterality: CMannerLaterality
    airstream: CMannerAirstream
    sibilance: CMannerSibilance
    vibrancy: CMannerVibrancy


# @convert(CManner)
@unique
class CMannerX(ContinuousEnum):  # FIXME(j_luo) fix parent class
    NONE = CMP('NONE', 'NONE', 'NONE', 'NONE', 'NONE', 'NONE')
    APPROXIMANT = CMP('APPROXIMANT', 'NON_NASAL', 'NON_LATERAL', 'PULMONIC_EGRESSIVE', 'NONE', 'NONE')
    CLICK = CMP('NONE', 'NON_NASAL', 'NON_LATERAL', 'LINGUAL_INGRESSIVE', 'NONE', 'NONE')
    # FIXME(j_luo) There might be some bug here at https://github.com/pettarin/ipapy/blob/4026e4e27e857820c12895a7d2e7281b1cff7723/ipapy/data/kirshenbaum.dat#L453,
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


@unique
class VHeightX(ContinuousEnum):
    NONE = auto()
    CLOSE = auto()
    NEAR_CLOSE = auto()
    CLOSE_MID = auto()
    MID = auto()
    OPEN_MID = auto()
    NEAR_OPEN = auto()
    OPEN = auto()


@unique
class VBacknessX(ContinuousEnum):
    NONE = auto()
    BACK = auto()
    NEAR_BACK = auto()
    CENTRAL = auto()
    NEAR_FRONT = auto()
    FRONT = auto()


@unique
class VRoundnessX(DiscreteEnum):
    NONE = auto()
    ROUNDED = auto()
    UNROUNDED = auto()
