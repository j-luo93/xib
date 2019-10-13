from arglib import Registry

reg = Registry('cfg')


@reg
class TestEn:
    lang = 'en'

    dim: int = 20
    num_features: int = 112
    num_feature_groups: int = 14
    check_interval: int = 50
    char_per_batch: int = 2000
    num_steps: int = 10000
    window_size: int = 3
    hidden_size: int = 100

    def __post_init__(self):
        self.data_path = f'data/phones_{self.lang}_idx.pth'


@reg
class TestEnP(TestEn):  # Only deal with ptype.
    mode: str = 'p'


@reg
class TestEnPCV(TestEn):
    mode: str = 'pcv'


@reg
class TestEnPDST(TestEn):
    mode: str = 'pdst'


_all_langs = [
    'sh', 'bg', 'fr', 'de', 'lt', 'pt', 'nl', 'ka', 'is', 'ro', 'fi', 'it', 'eo',
    'el', 'cs', 'syc', 'ga', 'ang', 'hy', 'cy', 'tr', 'ms', 'ady', 'sk', 'da', 'fa',
    'gem-pro', 'sl', 'lb', 'es', 'nci', 'gl', 'fo', 'enm', 'io', 'dsb', 'ba', 'tlh',
    'sv', 'no', 'tl', 'la', 'jbo', 'arc', 'he', 'sq', 'ps', 'nn', 'az', 'sga', 'sco', 'yue'
]

for lang in _all_langs:
    cap_lang = lang[0].upper() + lang[1:]
    new_cls = type(f'Test{cap_lang}PCV', (TestEnPCV,), {'lang': lang})
    reg(new_cls)
