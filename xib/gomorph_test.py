from pathlib import Path

import pandas as pd

from dev_misc import TestCase, test_with_arguments

from .gomorph import MorphClassRegistry


class TestGoMorph(TestCase):

    def setUp(self):
        super().setUp()
        folder = Path('./data/wulfila/')
        dfs = dict()
        for csv_path in folder.glob('*.csv'):
            name = csv_path.stem
            df = pd.read_csv(csv_path)
            # Use nullable dtype.
            col_to_change = list()
            for col in df.columns:
                if pd.api.types.is_integer_dtype(df[col].dtype):
                    col_to_change.append(col)
            df = df.astype({col: 'Int64' for col in col_to_change})
            dfs[name] = df

        def clean_df(df, cols_to_keep, **rename_dict):
            """Only keep certain columns and rename some others."""
            df = df[cols_to_keep]
            df = df.rename(columns=rename_dict)
            return df

        lemma = dfs['Lemmata']
        self.lemma = clean_df(lemma, ['ID', 'Lemma', 'Morphology'],
                              ID='LemmaID', Morphology='GomorphID')

    def test_all(self):
        mcr = MorphClassRegistry()
        for idx, mc in mcr.items():
            for l in self.lemma[self.lemma.GomorphID == idx].Lemma:
                mc(l)
