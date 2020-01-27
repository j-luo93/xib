from dataclasses import dataclass

from dev_misc.arglib import Registry
from dev_misc.utils import buggy

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


def add_langs(cls_prefix, base_cls):
    for lang in _all_other_langs:
        cap_lang = lang[0].upper() + lang[1:]
        new_cls = type(f'{cls_prefix}{cap_lang}', (base_cls,), {'lang': lang})
        reg(new_cls)


add_langs('LM', LMEn)


@reg
class CbowEn(LMEn):
    task: str = 'cbow'


add_langs('Cbow', CbowEn)


@reg
class AdaptLMEn(LMEn):
    task: str = 'adapt_lm'
    dense_input: bool = True
    learning_rate: float = 0.02
    num_steps: int = 1000


add_langs('AdaptLM', AdaptLMEn)


@reg
class AdaptCbowEn(AdaptLMEn):
    task: str = 'adapt_cbow'


add_langs('AdaptCbow', AdaptCbowEn)


@reg
class DecipherEn(LMEn):
    task: str = 'decipher'
    learning_rate: float = 5e-4


@reg
class DecipherEnTest(DecipherEn):

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/decipher_en_test.pth'


@reg
class DecipherEsTest(DecipherEn):
    lang = 'es'

    supervised: bool = True
    dev_data_path: str = 'data/Spanish.ipa.dev'

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.ipa.5000'


@reg
class DecipherEsNoisyP3Test(DecipherEsTest):
    dev_data_path: str = 'data/Spanish.ipa.noise_7500.dev'

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.ipa.noise_7500.5000'


@reg
class DecipherEsNoisyP5Test(DecipherEsTest):
    dev_data_path: str = 'data/Spanish.ipa.noise_12500.dev'

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.ipa.noise_12500.5000'


@buggy
@reg
class DecipherEsNoisyItalianP5Test(DecipherEsTest):
    dev_data_path: str = 'data/Spanish.Italian_p5.ipa.dev'

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.Italian_p5.ipa.5000'


@reg
class DecipherEsNoisyGermanP5Test(DecipherEsTest):
    dev_data_path: str = 'data/Spanish.German_p5.ipa.dev'

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.ipa.noise_12500.5000'


@reg
class DecipherEsToEsWithP5GermanTest(DecipherEsNoisyGermanP5Test):

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.ipa.5000'


@reg
class DecipherEsWithP5NoiseToEsWithP5GermanTest(DecipherEsNoisyGermanP5Test):

    def __post_init__(self):
        super().__post_init__()
        self.data_path = 'data/Spanish.ipa.noise_12500.5000'


@reg
class DecipherEsWithP5GermanTest(DecipherEsNoisyGermanP5Test):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = 'data/Spanish.German_p5.ipa.dev'


@reg
class ExtractEsWithP5GermanTest(DecipherEsWithP5GermanTest):
    dim: int = 30
    num_heads: int = 6
    vocab_path: str = 'data/Spanish.ipa.5000.words'
    optim_cls: str = 'sgd'
    learning_rate: float = 1.0
    max_segment_length: int = 20
    char_per_batch: int = 100
    eval_interval: int = 500
    task: str = 'extract'
    init_threshold: float = 10.0
    dense_input: bool = True
    check_interval: int = 50

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = 'data/tmp'


@reg
class SanityCheck(ExtractEsWithP5GermanTest):
    input_format: str = 'text'
    g2p_window_size: int = 1
    dim: int = 60
    optim_cls: str = 'sgd'
    init_ins_del: int = 100
    min_word_length: int = 4
    init_threshold: float = 30.0
    min_threshold: float = 0.99
    anneal_factor: float = 0.8
    num_steps: int = 1000
    reg_hyper: float = 1.0
    learning_rate: float = 0.1


@reg
class GotDeIpaMatched(SanityCheck):
    aligned: bool = True
    input_format: str = 'ipa'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got_de.shuf.ipa.100'
        self.vocab_path: str = './data/de.matched.words'


@reg
class GotDeIpaAligned(GotDeIpaMatched):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got-nhd.shuf.cog.200'
        self.vocab_path: str = './data/nhd.matched.words'


@reg
class GotDeIpaAlignedReverseGvs(GotDeIpaMatched):
    init_ins_del_cost: float = 3.5
    context_weight: float = 0.4
    g2p_window_size: int = 3
    char_per_batch: int = 400
    max_segment_length: int = 40
    postprocess_mode: str = 'gvs'
    lost_lang: str = 'got'
    known_lang: str = 'nhd'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got-nhd.corpus.100.tsv'
        self.vocab_path: str = './data/nhd.matched.100.words'


@reg
class GotDeIpaAlignedReverseGvsPrepare(GotDeIpaAlignedReverseGvs):
    task: str = 'prepare'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got-nhd.cog'


@reg
class FixGotGermOracleSpan(GotDeIpaAlignedReverseGvs):
    input_format: str = 'text'
    postprocess_mode: str = 'none'
    char_per_batch: int = 1000
    span_candidates: str = 'oracle_word'
    known_lang: str = 'germ'
    reg_hyper: float = 0.0
    context_weight: float = 0.5
    learning_rate: float = 0.05
    one2two: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got_lemma-germ_no_pref.all.corpus.tsv'
        self.vocab_path: str = './data/germ_no_pref.lemma.words'


@reg
class FixGotGermLemmaOnly(FixGotGermOracleSpan):
    span_candidates: str = 'all'
    char_per_batch: int = 400
    reg_hyper: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got-germ.lemma.uniq.tsv'


@reg
class FixGotNhdWithStemSmall(FixGotGermLemmaOnly):
    known_lang: str = 'nhd'
    span_candidates: str = 'all'
    char_per_batch: int = 400
    reg_hyper: float = 1.0
    use_stem: bool = True
    min_word_length: int = 3

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.small.tsv'
        self.vocab_path: str = './data/wulfila/processed/nhd.small.matched.stems'


@reg
class FixGotGermWithStemSmall(FixGotNhdWithStemSmall):
    known_lang: str = 'germ'
    char_per_batch: int = 320
    context_agg_mode: str = 'log_interpolation'
    max_segment_length: int = 40
    message: str = 'ml40'
    min_word_length: int = 4
    freq_hack: bool = True  # FIXME(j_luo) This should be removed.
    one2two: str = True
    init_ins_del_cost: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.small.got-germ.tsv'
        self.vocab_path: str = './data/wulfila/processed/germ.small.matched.stems'


@reg
class FixGotGermWithStemSmallCtcEmbedding(FixGotGermWithStemSmall):
    char_per_batch: int = 1000
    sort_by_length: bool = False
    use_s_prior: bool = True
    context_weight: float = 0.2
    use_ctc: bool = True
    use_base_embedding: bool = True
    dense_embedding: bool = True
    context_agg_mode: str = 'linear_interpolation'
    dim: int = 30
    reg_hyper: float = 0.1
    use_s_prior: bool = True
    dropout: float = 0.3
    learning_rate: float = 0.3
    save_alignment: bool = True


@reg
class FixGotAEOracleSpan(FixGotGermOracleSpan):
    known_lang: str = 'ae'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got_lemma-ae_no_pref.all.corpus.tsv'
        self.vocab_path: str = './data/ae_no_pref.lemma.words'


@reg
class FixGotNhdOracleSpan(FixGotGermOracleSpan):
    known_lang: str = 'nhd'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got_lemma-nhd_no_pref.all.corpus.tsv'
        self.vocab_path: str = './data/nhd_no_pref.lemma.words'
