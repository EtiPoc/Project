import pandas as pd
import preprocessor


pros = pd.read_csv('training_data_pro.csv')
pros.rename(columns={'0':'number'}, inplace=True)
lig = pd.read_csv('training_data_lig.csv')
lig.rename(columns={'0':'number'}, inplace=True)


class pipeline:
    def __init__(self, pros_df, lig_df):
        self.pros = pros_df[:100]
        self.lig = lig_df[:100]
        self.df = pd.DataFrame

    def construct_df(self):
        self.pros['key'] = 0
        self.lig['key'] = 0
        self.df = pd.merge(self.pros, self.lig, on='key', suffixes=('_pro', '_lig'))
        self.df.key = self.df.number_pro == self.df.number_lig
        self.df.rename(columns={'key': 'label'}, inplace=True)


test_pipeline = pipeline(pros, lig)
test_pipeline.construct_df()
features = test_pipeline.df.apply(lambda row: preprocessor.Sample(row).features.loc[0], 1)
