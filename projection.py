import pandas as pd
import numpy as np
import ast

pros = pd.read_csv('training_data_pro.csv')
pros.rename(columns={'0': 'number'}, inplace=True)
ligs = pd.read_csv('training_data_lig.csv')
ligs.rename(columns={'0': 'number'}, inplace=True)

pro = pros.loc[0]
lig = ligs.loc[0]


class Projector:
    def __init__(self, molecule, size):
        X = molecule.X_list
        Y = molecule.Y_list
        Z = molecule.Z_list
        self.size = size
        self.molecule = pd.DataFrame(columns=[i + '_coord' for i in ['X', 'Y', 'Z']])
        self.molecule.X_coord = ast.literal_eval(X)
        self.molecule.Y_coord = ast.literal_eval(Y)
        self.molecule.Z_coord = ast.literal_eval(Z)

    def scale(self):
        x_min = self.molecule.X_coord.min()
        self.molecule.X_coord = self.molecule.X_coord - x_min
        y_min = self.molecule.Y_coord.min()
        self.molecule.Y_coord = self.molecule.Y_coord - y_min
        z_min = self.molecule.Z_coord.min()
        self.molecule.Z_coord = self.molecule.Z_coord - z_min

    def convert(self):
        x_max = self.molecule.X_coord.max()
        self.molecule.X_coord = self.molecule.apply(lambda row: int((row.X_coord/x_max)*self.size), 1)
        y_max = self.molecule.Y_coord.max()
        self.molecule.Y_coord = self.molecule.apply(lambda row: int((row.Y_coord/y_max)*self.size), 1)
        z_max = self.molecule.Z_coord.max()
        self.molecule.Z_coord = self.molecule.apply(lambda row: int((row.Z_coord/z_max)*self.size), 1)


self = Projector(pro, 64)
