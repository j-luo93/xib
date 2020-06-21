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
class ExtractEsWithP5GermanTest(SharedConfig):
    dim: int = 30
    vocab_path: str = 'data/Spanish.ipa.5000.words'
    optim_cls: str = 'sgd'
    learning_rate: float = 1.0
    char_per_batch: int = 100
    eval_interval: int = 500
    dense_input: bool = True
    check_interval: int = 50

    def __post_init__(self):
        self.data_path: str = 'data/tmp'


@reg
class SanityCheck(ExtractEsWithP5GermanTest):
    input_format: str = 'text'
    g2p_window_size: int = 1
    dim: int = 60
    optim_cls: str = 'sgd'
    init_ins_del: int = 100
    min_word_length: int = 4
    anneal_factor: float = 0.8
    num_steps: int = 1000
    reg_hyper: float = 1.0
    learning_rate: float = 0.1


@reg
class GotDeIpaMatched(SanityCheck):
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
    lost_lang: str = 'got'
    known_lang: str = 'nhd'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/got-nhd.corpus.100.tsv'
        self.vocab_path: str = './data/nhd.matched.100.words'


@reg
class FixGotGermOracleSpan(GotDeIpaAlignedReverseGvs):
    input_format: str = 'text'
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
    context_weight: float = 0.2
    use_base_embedding: bool = True
    dense_embedding: bool = True
    # context_agg_mode: str = 'linear_interpolation'
    dim: int = 30
    reg_hyper: float = 0.1
    dropout: float = 0.3
    # learning_rate: float = 0.3
    save_alignment: bool = True
    min_word_length: int = 3
    learning_rate: float = 0.1
    context_agg_mode: str = 'log_interpolation'


@reg
class FixSanityCheck(FixGotGermWithStemSmallCtcEmbedding):
    pr_hyper: float = 0.0
    main_loss_hyper: float = 0.0
    save_alignment: bool = True
    eval_interval: int = 250
    num_steps: int = 500
    l_pr_hyper: float = 10.0
    bias_mode: str = 'learned'
    check_interval: int = 10
    learning_rate: float = 0.1
    reg_hyper: float = 0.01


@reg
class LTSanityCheck(FixSanityCheck):
    baseline: float = 0.05
    bias_mode: str = 'fixed'
    reg_hyper: float = 0.1
    expected_ratio: float = 0.7
    char_per_batch: int = 400
    l_pr_hyper: float = 1.0
    pr_hyper: float = 10.0
    init_expected_ratio: float = 1.0
    random_seed: int = 74823
    check_interval: int = 25
    inference_mode: str = 'old'
    reward_mode: str = 'ln_div'


@reg
class NoContextSanityCheck(LTSanityCheck):
    pr_hyper: float = 30.0
    min_segment_length: int = 30
    reward_mode: str = 'div'
    freq_hack: bool = False
    downsample: bool = True
    num_steps: int = 1000
    main_loss_hyper: float = 0.0
    context_weight: float = 0.0
    reg_hyper: float = 0.1
    learning_rate: float = 0.06
    aligner_lr: float = 0.06
    bij_reg: float = 12.0
    emb_norm: float = 1.0
    ent_reg: float = 0.06
    l_pr_hyper: float = 6.0
    max_grad_norm: float = 5.0
    num_rounds: int = 1
    use_entropy_reg: bool = True
    temperature: float = 0.2


@reg
class VowelCheck(NoContextSanityCheck):
    emb_norm: float = 1.0
    temperature: float = 0.2
    reward_mode: str = 'thresh'
    pr_hyper: float = 10.0
    baseline: float = 0.0
    init_baseline: float = 0.0
    max_baseline: float = 0.0
    anneal_baseline: bool = True
    unit_aligner_init_mode: str = 'zero'
    init_interval: float = 0.05
    normalize: float = 0.0
    optim_cls: str = 'sgd'
    learning_rate: float = 0.2
    aligner_lr: float = 0.2
    dim: int = 70
    dropout: float = 0.5
    num_steps: int = 2000
    check_interval: int = 10
    bij_reg_hyper: float = 100.0
    ent_reg_hyper: float = 0.1
    feat_aligner_init_mode: str = 'zero'
    mean_mode: str = 'char'
    l_pr_hyper: float = 10.0
    min_segment_length: int = 3
    max_grad_norm: float = 1.0
    anneal_temperature: bool = True
    init_temperature: float = 0.3
    end_temperature: float = 0.1
    anneal_er: bool = True
    pr_mode: str = 'barrier'
    inference_mode: str = 'old'
    expected_ratio: float = 0.5


@reg
class FinalPgm(VowelCheck):
    # context_weight: float = 0.2
    context_weight: float = 0.0
    known_lang: str = 'pgm'
    lost_lang: str = 'got'
    num_steps: int = 3000
    dim: int = 100
    unit_aligner_init_mode: str = 'uniform'
    use_entropy_reg: bool = False
    use_new_model: bool = True
    init_temperature: float = 0.2
    end_temperature: float = 0.2
    init_expected_ratio: float = 1.0
    expected_ratio: float = 0.5
    init_ins_del_cost: float = 0.0
    min_ins_del_cost: float = 3.5
    min_word_length: int = 4
    min_segment_length: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.got-germ.trunc.tsv'
        self.vocab_path: str = './data/wulfila/processed/germ.matched.trunc.stems'


@reg
class FinalNon(FinalPgm):

    # init_ins_del_cost: float = 10.0
    known_lang: str = 'non'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.got-non.trunc.tsv'
        self.vocab_path: str = './data/wulfila/processed/non.matched.trunc.stems'


@reg
class FinalAng(FinalPgm):

    # init_ins_del_cost: float = 10.0
    known_lang: str = 'ang'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.got-ang.trunc.tsv'
        self.vocab_path: str = './data/wulfila/processed/ang.matched.trunc.stems'


@reg
class FinalGothicPgm100(FinalPgm):

    segmented: bool = True
    downsample: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.got-germ.trunc.tsv'
        self.vocab_path: str = './data/wulfila/processed/germ.matched.trunc.stems'


@reg
class FinalGothicNon100(FinalGothicPgm100):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.got-non.trunc.tsv'
        self.vocab_path: str = './data/wulfila/processed/non.matched.trunc.stems'


@reg
class FinalGothicAng100(FinalGothicPgm100):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.got-ang.trunc.tsv'
        self.vocab_path: str = './data/wulfila/processed/ang.matched.trunc.stems'


@reg
class FinalContrastLatin(FinalPgm):

    known_lang: str = 'ang'
    use_oracle: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.got-germ.trunc.seg.p5.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.lat.stems'


@reg
class FinalContrastSpanish(FinalContrastLatin):

    def __post_init__(self):
        super().__post_init__()
        self.vocab_path: str = './data/wulfila/processed/contrast.es.stems'


@reg
class FinalContrastTurkish(FinalContrastLatin):

    def __post_init__(self):
        super().__post_init__()
        self.vocab_path: str = './data/wulfila/processed/contrast.tur.stems'


@reg
class FinalContrastIndo(FinalContrastLatin):

    def __post_init__(self):
        super().__post_init__()
        self.vocab_path: str = './data/wulfila/processed/contrast.ind.stems'


@reg
class FinalContrastHungarian(FinalContrastLatin):

    def __post_init__(self):
        super().__post_init__()
        self.vocab_path: str = './data/wulfila/processed/contrast.hun.stems'


@reg
class FinalContrastBasque(FinalContrastLatin):

    def __post_init__(self):
        super().__post_init__()
        self.vocab_path: str = './data/Iberian/Basque/eu.all.stems'


@reg
class FinalUgaReal(FinalPgm):
    min_word_length: int = 3
    min_segment_length: int = 3
    span_candidates: str = 'oracle_word'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/uga.real.small.tsv'
        self.vocab_path: str = './data/heb.real.stems'


@reg
class FinalUga(FinalUgaReal):
    use_base_embedding: bool = True
    dense_embedding: bool = False
    base_embedding_dim: int = 490

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/uga.real.small.tsv'
        self.vocab_path: str = './data/heb.real.stems'


@reg
class FinalXib(FinalPgm):
    expected_ratio: float = 0.1
    min_word_length: int = 3
    min_segment_length: int = 3


@reg
class FinalXibBasquePos(FinalXib):
    use_oracle: bool = True
    known_lang: str = 'lat'
    min_word_length: int = 4
    min_segment_length: int = 4

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/Iberian/Basque/eu.lat.stems'


@reg
class FinalXibBasqueNeg(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/Iberian/Basque/eu.unk.stems'


@reg
class FinalXibContrastLatin(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.lat.stems'


@reg
class FinalXibContrastTurkish(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.tur.stems'


@reg
class FinalXibContrastHungarian(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.hun.stems'


@reg
class FinalXibContrastProtoGermanic(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/wulfila/processed/germ.matched.trunc.stems'


@reg
class FinalXibContrastSpanish(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.es.stems'


@reg
class FinalXibContrastOldNorse(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/wulfila/processed/non.matched.trunc.stems'


@reg
class FinalXibContrastOldEnglish(FinalXibBasquePos):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
        self.vocab_path: str = './data/wulfila/processed/ang.matched.trunc.stems'


@reg
class FinalUgaContrastArabic(FinalUgaReal):

    min_word_length: int = 3
    min_segment_length: int = 3
    span_candidates: str = 'oracle_word'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/uga.real.small.tsv'
        self.vocab_path: str = './data/uga/contrast.ara.stems'


@reg
class FinalUgaContrastLatin(FinalUgaContrastArabic):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/uga.real.small.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.lat.stems'


@reg
class FinalUgaContrastHungarian(FinalUgaContrastArabic):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/uga.real.small.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.hun.stems'


@reg
class FinalUgaContrastSpanish(FinalUgaContrastArabic):

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/uga.real.small.tsv'
        self.vocab_path: str = './data/wulfila/processed/contrast.es.stems'


# @reg
# class FinalXibContrastOldEnglish(FinalXibBasquePos):

#     def __post_init__(self):
#         super().__post_init__()
#         self.data_path: str = './data/Iberian/corpus.xib-lat.tsv'
#         self.vocab_path: str = './data/wulfila/processed/ang.matched.trunc.stems'


@reg
class FixSanityCheckInit(FixSanityCheck):
    baseline: float = 0.05
    bias_mode: str = 'fixed'
    reg_hyper: float = 0.1
    expected_ratio: float = 0.7
    init_expected_ratio: float = 1.0
    char_per_batch: int = 400
    l_pr_hyper: float = 1.0
    pr_hyper: float = 20.0
    check_interval: int = 25


@reg
class FixGotAngWithStemSmallCtcEmbedding(FixGotGermWithStemSmallCtcEmbedding):
    known_lang: str = 'ang'

    def __post_init__(self):
        super().__post_init__()
        self.data_path: str = './data/wulfila/processed/corpus.small.got-ang.tsv'
        self.vocab_path: str = './data/wulfila/processed/ang.small.matched.stems'


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
