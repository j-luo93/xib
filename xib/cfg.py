from arglib import Registry

reg = Registry('cfg')


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
    hidden_size: int = 100


@reg
class TestEnP(TestEn):  # Only deal with ptype.
    mode: str = 'p'


@reg
class TestEnPCV(TestEn):
    mode: str = 'pcv'


@reg
class TestEnPDST(TestEn):
    mode: str = 'pdst'
