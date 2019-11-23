from dataclasses import dataclass

from dev_misc.arglib import Registry

reg = Registry('cfg')


@dataclass
class SharedConfig:
    hidden_size: int = 100
    num_feature_groups: int = 14
    feat_groups: str = 'pcv'


@reg
class LMEn(SharedConfig):
    lang = 'en'

    dim: int = 20
    num_features: int = 112
    check_interval: int = 50
    char_per_batch: int = 2000
    num_steps: int = 10000
    window_size: int = 5
    use_cached_pth: bool = True

    def __post_init__(self):
        self.data_path = f'data/phones_{self.lang}_idx.pth'


@reg
class LMEnP(LMEn):  # Only deal with ptype.
    feat_groups: str = 'p'


@reg
class LMEnPDST(LMEn):
    feat_groups: str = 'pdst'


_all_other_langs = {
    'sh', 'bg', 'fr', 'de', 'lt', 'pt', 'nl', 'ka', 'is', 'ro', 'fi', 'it', 'eo',
    'el', 'cs', 'syc', 'ga', 'ang', 'hy', 'cy', 'tr', 'ms', 'ady', 'sk', 'da', 'fa',
    'gem-pro', 'sl', 'lb', 'es', 'nci', 'gl', 'fo', 'enm', 'io', 'dsb', 'ba', 'tlh',
    'sv', 'no', 'tl', 'la', 'jbo', 'arc', 'he', 'sq', 'ps', 'nn', 'az', 'sga', 'sco',
    'xib', 'yue', 'eu', 'xaq'
}

for lang in _all_other_langs:
    cap_lang = lang[0].upper() + lang[1:]
    new_cls = type(f'LM{cap_lang}', (LMEn,), {'lang': lang})
    reg(new_cls)


@reg
class AdaptLMEn(LMEn):
    task: str = 'adapt'
    dense_input: bool = True
    learning_rate: float = 0.02
    num_steps: int = 1000


for lang in _all_other_langs:
    cap_lang = lang[0].upper() + lang[1:]
    new_cls = type(f'AdaptLM{cap_lang}', (AdaptLMEn, ), {'lang': lang})
    reg(new_cls)


@reg
class DecipherEn(LMEn):
    task: str = 'decipher'
    mode: str = 'local-supervised'
    learning_rate: float = 5e-4
    use_cached_pth: bool = False


@reg
class DecipherEnTest(DecipherEn):

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/decipher_en_test.pth'


@reg
class DecipherEsTest(DecipherEn):
    lang = 'es'

    supervised: bool = True
    dev_data_path: str = 'data/Spanish.clean.100.dev.ipa'

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.clean.100.ipa'


@reg
class MetricPCV(SharedConfig):
    num_epochs: int = 500
    eval_interval: int = 10
    check_interval: int = 10
    task: str = 'metric'
    family_file_path: str = 'data/families.txt'
    data_path: str = 'data/direct_transfer.tsv'
    num_lang_pairs: int = 100
    learning_rate: float = 0.02
    hidden_size: int = 50
