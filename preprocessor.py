import pandas as pd
import ast
import numpy as np

sample_df = pd.read_csv("sample_df.csv")


def distance(row):
    return np.sqrt((row['x_prot']-row['x_lig'])**2 + (row['y_prot']-row['y_lig'])**2 + (row['z_prot']-row['z_lig'])**2)


class Sample:

    def __init__(self, sample):
        self.prot = self.prot(sample)
        self.lig = self.lig(sample)
        self.comb = self.prot.merge(self.lig, on='key')
        self.distances()
        self.features = pd.DataFrame([0])
        self.compute_features()
        self.features.drop(0, 1, inplace=True)

    def distances(self):
        self.comb['distance'] = self.comb.apply(lambda row: distance(row), 1)

    def min_distances(self):
        new_features = []
        distances = self.comb.groupby(['x_lig', 'y_lig', 'z_lig']).distance
        new_features += [distances.min().mean()]
        new_features += [distances.min().min()]
        new_features += [distances.min().std()]
        polars_in_min = (self.comb.loc[distances.idxmin()][['atomtype_list_pro', 'atomtype_list_lig']]!='h').sum().sum()
        new_features += [polars_in_min]
        self.features = self.features.join(pd.DataFrame([new_features], columns=['avg_min_distance', 'min_min_distance', 'std_min_distances', 'polars_in_min']))

    def polar_min_distances(self):
        polar_prot = self.comb[self.comb['atomtype_list_pro'] != 'h'].groupby(['x_prot', 'y_prot', 'z_prot']).distance.min()
        polar_lig = self.comb[self.comb['atomtype_list_lig'] != 'h'].groupby(['x_lig', 'y_lig', 'z_lig']).distance.min()
        both = polar_prot.append(polar_lig)
        num_prot = polar_prot.shape[0]
        num_lig = polar_lig.shape[0]
        min = np.nanmin((polar_prot.min(), polar_lig.min()))
        avg = both.mean()
        new_features = [num_prot, num_lig, min, avg]
        self.features = self.features.join(pd.DataFrame([new_features], columns=['num_polar_prot', 'num_polar_lig',
                                                                                           'min_dist_polar', 'avg_dist_polar']))

    def compute_features(self):
        self.min_distances()
        self.polar_min_distances()

    def prot(self, data):
        df = pd.DataFrame()
        df = df.append(pd.DataFrame(ast.literal_eval(data.X_list_pro), columns=['x_prot']))
        df = df.join(pd.DataFrame(ast.literal_eval(data.Y_list_pro), columns=['y_prot']))
        df = df.join(pd.DataFrame(ast.literal_eval(data.Z_list_pro), columns=['z_prot']))
        df = df.join(pd.DataFrame(ast.literal_eval(data.atomtype_list_pro), columns=['atomtype_list_pro']))
        df['key'] = 0
        return df

    def lig(self, data):
        df = pd.DataFrame()
        df = df.append(pd.DataFrame(ast.literal_eval(data.X_list_lig), columns=['x_lig']))
        df = df.join(pd.DataFrame(ast.literal_eval(data.Y_list_lig), columns=['y_lig']))
        df = df.join(pd.DataFrame(ast.literal_eval(data.Z_list_lig), columns=['z_lig']))
        df = df.join(pd.DataFrame(ast.literal_eval(data.atomtype_list_lig), columns=['atomtype_list_lig']))
        df['key'] = 0
        return df



# data = sample_df.loc[100]
# sample = Sample(data)
sample_df = sample_df.loc[:10]
features = sample_df.apply(lambda row: row.append(Sample(row).features.loc[0]), 1)
