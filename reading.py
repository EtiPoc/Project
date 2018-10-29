import numpy as np
import pandas as pd
import csv



def read_pdb(filename):
    with open(filename, 'r') as file:
        strline_L = file.readlines()
    # print(strline_L)

    X_list = list()
    Y_list = list()
    Z_list = list()
    atomtype_list = list()
    for strline in strline_L:
        # removes all whitespace at the start and end, including spaces, tabs, newlines and carriage returns
        stripped_line = strline.strip()

        line_length = len(stripped_line)
        # print("Line length:{}".format(line_length))
        if line_length < 78:
            print("ERROR: line length is different. Expected>=78, current={}".format(line_length))

        X_list.append(float(stripped_line[30:38].strip()))
        Y_list.append(float(stripped_line[38:46].strip()))
        Z_list.append(float(stripped_line[46:54].strip()))

        atomtype = stripped_line[76:78].strip()
        if atomtype == 'C':
            atomtype_list.append('h') # 'h' means hydrophobic
        else:
            atomtype_list.append('p') # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


# array : ([lig : ['X_list', 'Y_list', 'Z_list', 'atomtype_list']], [prot : same as lig], 1 if match 0 else)

def create_df(files_nb, ratio=1):
    """ratio is the number of negative sample for each positive one
read through all data files and output a DataFrame with the indexes and data of each pair (all the positives and the negatives)"""
    data_list = pd.DataFrame()

    for i in range(1, files_nb+1):

        number = (4-len(str(i)))*"0" + str(i)

        filename_lig = "training_data/"+number+"_lig_cg.pdb"
        X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig = read_pdb(filename_lig)

        filename_pro = "training_data/"+number+"_pro_cg.pdb"
        X_list_pro, Y_list_pro, Z_list_pro, atomtype_list_pro = read_pdb(filename_pro)

        data_list = data_list.append([[[X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig], [X_list_pro, Y_list_pro, Z_list_pro, atomtype_list_pro], 1]])

    for i in range(1, ratio*(files_nb+1)):
        rand = np.random.randint(files_nb, size=2) + 1
        while rand[0] == rand[1]:
            rand = np.random.randint(files_nb, size=2) + 1

        number_lig = (4-len(str(rand[0])))*"0" + str(rand[0])
        filename_lig = "training_data/"+number_lig+"_lig_cg.pdb"
        X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig = read_pdb(filename_lig)
        number_pro = (4-len(str(rand[1])))*"0" + str(rand[1])
        filename_pro = "training_data/"+number_pro+"_pro_cg.pdb"
        X_list_pro, Y_list_pro, Z_list_pro, atomtype_list_pro = read_pdb(filename_pro)
        data_list = data_list.append([[[X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig], [X_list_pro, Y_list_pro, Z_list_pro, atomtype_list_pro], 0]])
    return data_list


def centroid(x_list, y_list, z_list):
    return [np.mean(x_list), np.mean(y_list), np.mean(z_list)]


class Box:
    """
    features = [nb of atoms in point, nb of lig in point, nb of polar in point, nb of polar-lig in point,
        nb of atoms in neighbors, nb of lig in neighbors, nb of polar in neighbors, nb of polar-lig in neighbors]
    """
    def __init__(self, lig_list, pro_list, size=20, step=1, features_size=8):
        self.size = size
        self.step = step
        self.center = centroid(lig_list[0], lig_list[1], lig_list[2])
        self.grid = np.zeros((size, size, size, features_size))
        self.lig_list = lig_list
        self.pro_list = pro_list

    def assign_point(self, X, Y, Z, atomtype, is_lig):

        if((X-self.center[0] > -self.size/2) and (X-self.center[0] < self.size/2)
                and (Y-self.center[1] > -self.size/2) and (Y-self.center[1] < self.size/2)
                and (Z-self.center[2] > -self.size/2) and (Z-self.center[2] < self.size/2)):
            x_grid = int((X-self.center[0])/self.step + self.size/2)
            y_grid = int((Y-self.center[1])/self.step + self.size/2)
            z_grid = int((Z-self.center[2])/self.step + self.size/2)
            self.grid[x_grid, y_grid, z_grid, 0] += 1

            if is_lig:
                self.grid[x_grid, y_grid, z_grid, 1] += 1

            if atomtype == 'p':
                self.grid[x_grid, y_grid, z_grid, 2] += 1

            if is_lig and atomtype == 'p':
                self.grid[x_grid, y_grid, z_grid, 3] += 1

    def fill_grid(self):
        for i in range(len(self.lig_list[0])):
            self.assign_point(self.lig_list[0][i], self.lig_list[1][i], self.lig_list[2][i], self.lig_list[3][i], 1)

        for i in range(len(self.pro_list[0])):
            self.assign_point(self.pro_list[0][i], self.pro_list[1][i], self.pro_list[2][i], self.pro_list[3][i], 0)

    def compute_neighbors_features(self):
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):

                    for i in [-1, 0, 1]:
                        for j in [-1, 0, 1]:
                            for k in [-1, 0, 1]:

                                if (i, j, k) != (0, 0, 0):
                                    if x+i >= 0 and x+i < self.size:
                                        if y + j >= 0 and y + j < self.size:
                                            if (y + k >= 0 and z + k < self.size):
                                                self.grid[x, y, z][4:8] += self.grid[x+i, y+j, z+k][0:4]

    def normalize_features(self):

        for i in range(self.grid[0, 0, 0]):
            max = max(abs(self.grid[:, :, :][i]))
            self.grid[:, :, :][i] /= max


if __name__ == "__main__":
    a = create_df(3000)
    grids = []
    y = []
    for i in range(a.shape[0]):
        if i % 10 == 0:
            print(i)
        b = Box(a.iloc[i, 0], a.iloc[i, 1], 10, 2)
        y += [a.iloc[i, 2]]
        b.fill_grid()
        b.compute_neighbors_features()
        grids += [b.grid]
    grids = np.array(grids)
    np.save('training_data.npy', grids)
    y = np.array(y)
    np.save('training_labels.npy', y)












