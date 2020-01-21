from typing import ClassVar, List, Set

import pandas as pd

PD_DF = pd.DataFrame


class BaseTable:

    _required_columns: ClassVar[List[str]] = None

    def __init__(self, data: PD_DF):
        self._check_columns(data)
        self.data = data
        self.postprocess()

    def _check_columns(self, data: PD_DF):
        cls = type(self)
        for col in cls._required_columns:
            if col not in data.columns:
                raise TypeError(f'Column {col} not present in the data frame which has columns {data.columns}.')

    def postprocess(self):
        pass


class MonoTable(BaseTable):

    def __init__(self, data: PD_DF, lang: str):
        super().__init__(data)
        self.lang = lang


class Tokens(MonoTable):
    """`Tokens` class defines the sequential structure of a corpus."""

    _required_columns = ['SegmentID', 'Position', 'Token', 'LemmaID']

    def postprocess(self):
        self.data['LemmaID'] = self.data['LemmaID'].astype('Int64')


class Lemmas(MonoTable):

    _required_columns = ['LemmaID', 'Lemma']

    def postprocess(self):
        self.data['LemmaID'] = self.data['LemmaID'].astype('Int64')


class Stems(MonoTable):
    """`Stems` class defines the mapping between tokens and stems."""

    _required_columns = ['Token', 'Stems']


class CogSet(BaseTable):

    _required_columns = ['CogID', 'Source', 'Lemma', 'Language']

    def __init__(self, data: PD_DF, source_lang: str):
        super().__init__(data)
        self.source_lang = source_lang


def generate_data_file(lost_tokens: Tokens,
                       lost_lemmas: Lemmas,
                       lost_stems: Stems,
                       known_stems: Stems,
                       cog_set: CogSet,
                       out_path: str) -> PD_DF:
    # Get some meta-data first.
    source_lang = cog_set.source_lang
    lost_lang = lost_tokens.lang
    known_lang = known_stems.lang
    assert lost_lemmas.lang == lost_stems.lang == lost_lang

    # Get stem info for each token in the corpus.
    ts = pd.merge(lost_tokens.data, lost_stems.data, left_on='Token', right_on='Token', how='left')
    # Add lemma information.
    tsl = pd.merge(ts, lost_lemmas.data, left_on='LemmaID', right_on='LemmaID', how='left')

    # Remove sentences that are not fully analyzed with lemmas.
    no_lemma_mask = tsl['LemmaID'].isnull()
    incomplete_segments = set(tsl[no_lemma_mask]['SegmentID'])
    total_segments = len(set(tsl['SegmentID']))
    print(f'Missing {no_lemma_mask.sum()}/{len(tsl)} lost lemmas in {len(incomplete_segments)}/{total_segments} segments.')

    # Get cognate alignments.
    def get_lemmas(lang: str, group) -> List[str]:
        if lang == source_lang:
            lemmas = group.Source.values
        else:
            lemmas = group[group.Language == lang].Lemma.values
        return sorted(set(lemmas))

    data = list()
    for cog_id, group in cog_set.data.groupby('CogID'):
        langs = set(group['Language'])
        langs.add(source_lang)
        if lost_lang in langs and known_lang in langs:
            lost_lemmas = get_lemmas(lost_lang, group)
            known_lemmas = get_lemmas(known_lang, group)
            data.append((lost_lemmas, known_lemmas))
    cog_df = pd.DataFrame(data, columns=['lost_lemma', 'known_lemma'])
    cog_df = cog_df.explode('lost_lemma')

    # cog_set_pt = cog_set.data.pivot_table(index=['CogID', 'Source'], columns='Language', values='Lemma', aggfunc=set)
    # cog_df = cog_set_pt.reset_index(level=1).rename(columns={'Source': 'gem-pro'})
    # cog_df[[lost_lang, known_lang]]

    # Add stem info on the known side.
    flat_cog_df = cog_df.explode('known_lemma')
    flat_cog_df = pd.merge(flat_cog_df, known_stems.data, left_on='known_lemma', right_on='Token', how='left')
    print(f'Missing {flat_cog_df.Stems.isnull().sum()} stems.')
    flat_cog_df = flat_cog_df.dropna(subset=['Stems'])
    # Combine stems and their original forms.
    cog_df = flat_cog_df.pivot_table(index='lost_lemma',
                                     values=['known_lemma', 'Stems'],
                                     aggfunc={
                                         'known_lemma': lambda lst: '|'.join(set(lst)),
                                         'Stems': lambda lst: '|'.join(set(lst))
                                     })

    # Match lost tokens and known cognates based on lemmas.
    matched = pd.merge(tsl, cog_df, left_on='Lemma', right_on='lost_lemma', how='left')

    # Write to file.
    matched.to_csv(out_path, sep='\t', index=None)

    return matched
