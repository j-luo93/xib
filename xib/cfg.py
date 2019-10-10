from enum import Enum, auto, unique
from arglib import Registry

reg = Registry('cfg')


@unique
class IPAFeature(Enum):
    NONE = 0
    VOICED = 1
    VOICELESS = 2
    ALVEOLAR = 3
    BILABIAL = 4
    VELAR = 5
    GLOTTAL = 6
    PALATAL = 7
    UVULAR = 8
    LABIO_DENTAL = 9
    PALATO_ALVEOLAR = 10
    LABIO_VELAR = 11
    RETROFLEX = 12
    DENTAL = 13
    LABIO_PALATAL = 14
    ALVEOLO_PALATAL = 15
    LABIO_ALVEOLAR = 16
    PHARYNGEAL = 17
    PALATO_ALVEOLO_VELAR = 18
    NASAL = 19
    FLAP = 20
    PLOSIVE = 21
    TRILL = 22
    NON_SIBILANT_FRICATIVE = 23
    LATERAL_APPROXIMANT = 24
    APPROXIMANT = 25
    SIBILANT_FRICATIVE = 26
    SIBILANT_AFFRICATE = 27
    CLICK = 28
    IMPLOSIVE = 29
    LATERAL_FRICATIVE = 30
    EJECTIVE = 31
    LATERAL_CLICK = 32
    LATERAL_AFFRICATE = 33
    NON_SIBILANT_AFFRICATE = 34
    LATERAL_FLAP = 35
    EJECTIVE_AFFRICATE = 36
    EJECTIVE_FRICATIVE = 37
    LATERAL_EJECTIVE_AFFRICATE = 38
    OPEN_MID = 39
    CLOSE_MID = 40
    OPEN = 41
    CLOSE = 42
    NEAR_OPEN = 43
    NEAR_CLOSE = 44
    MID = 45
    BACK = 46
    FRONT = 47
    CENTRAL = 48
    NEAR_FRONT = 49
    NEAR_BACK = 50
    ROUNDED = 51
    UNROUNDED = 52
    ASPIRATED = 53
    LABIALIZED = 54
    NASALIZED = 55
    SYLLABIC = 56
    NO_AUDIBLE_RELEASE = 57
    RETRACTED_TONGUE_ROOT = 58
    NON_SYLLABIC = 59
    PALATALIZED = 60
    BREATHY_VOICED = 61
    VELARIZED = 62
    CREAKY_VOICED = 63
    LOWERED = 64
    PHARYNGEALIZED = 65
    RAISED = 66
    ADVANCED = 67
    CENTRALIZED = 68
    TIE_BAR_ABOVE = 69
    RHOTACIZED = 70
    RETRACTED = 71
    LESS_ROUNDED = 72
    TIE_BAR_BELOW = 73
    APICAL = 74
    LAMINAL = 75
    LATERAL_RELEASE = 76
    ADVANCED_TONGUE_ROOT = 77
    MORE_ROUNDED = 78
    PRIMARY_STRESS = 79
    LONG = 80
    EXTRA_SHORT = 81
    HALF_LONG = 82
    WORD_BREAK = 83
    LINKING = 84
    LOW_LEVEL = 85
    HIGH_LEVEL = 86
    EXTRA_LOW_LEVEL = 87
    EXTRA_HIGH_LEVEL = 88
    MID_LEVEL = 89
    FALLING_CONTOUR = 90
    RISING_CONTOUR = 91
    RISING_FALLING_CONTOUR = 92
    HIGH_MID_FALLING_CONTOUR = 93
    LOW_RISING_CONTOUR = 94
    HIGH_RISING_CONTOUR = 95
    MID_LOW_FALLING_CONTOUR = 96
    DOWNSTEP = 97


FEAT_COLS = ['c_voicing', 'c_place', 'c_manner', 'v_height', 'v_backness', 'v_roundness',
             'diacritics', 's_stress', 's_length', 's_break', 't_level', 't_contour', 't_global']


@reg
class TestEn:
    data_path = 'data/phones_en.pth'
    dim: int = 250
    num_features: int = 98
    num_feature_groups: int = len(FEAT_COLS)
