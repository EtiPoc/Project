import sys
import pandas as pd
import keras
from reading import  Box
from testing_data_release.read_testing_pdb_file import read_pdb


def create_test_df(files_nb=824):
    """ read all the test files and return one data set for the ligands and one dataset for the proteins"""

    data_pro = pd.DataFrame()
    data_lig = pd.DataFrame()
    for i in range(1, files_nb + 1):
        number = (4 - len(str(i))) * "0" + str(i)

        filename_lig = "testing_data_release/testing_data/" + number + "_lig_cg.pdb"
        X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig = read_pdb(filename_lig)

        filename_pro = "testing_data_release/testing_data/" + number + "_pro_cg.pdb"
        X_list_pro, Y_list_pro, Z_list_pro, atomtype_list_pro = read_pdb(filename_pro)

        data_lig = data_lig.append(pd.DataFrame([[i, [X_list_lig, Y_list_lig, Z_list_lig, atomtype_list_lig]]], columns=['index_lig', 'data_lig']))
        data_pro = data_pro.append(pd.DataFrame([[i, [X_list_pro, Y_list_pro, Z_list_pro, atomtype_list_pro]]], columns=['index_pro', 'data_pro']))

    data_lig.reset_index()
    data_pro.reset_index()

    return data_lig, data_pro


def process_test_df(data_pro, data_lig, model):
    """ for every possible combination of protein/ligand
        performs the prediction with the trained model
        returns them as a dataframe"""
    data_list = pd.DataFrame(columns=['idx_pro', 'idx_lig', 'pred'])
    for i in range(data_pro.shape[0]):
        if i % 10 == 0:
            print(i)
        for j in range(data_lig.shape[0]):
            if j % 100 == 0:
                print(i, j)
            box = Box(data_lig.iloc[i, 1], data_pro.iloc[i, 1], 10, 2)
            box.fill_grid()
            box.compute_neighbors_features()
            prediction = model.predict(box.grid.reshape((1, 10, 10, 10, 8)))[0][0]
            data_list = data_list.append(pd.DataFrame([[i, j, prediction]], columns=['idx_pro', 'idx_lig', 'pred']))
    return data_list


def find_best_pairs(predictions):
    """for each proteins, find the 10 ligands with the highest binding probability"""

    cols = ['pro_id', 'lig1_id', 'lig2_id', 'lig3_id', 'lig4_id', 'lig5_id', 'lig6_id', 'lig7_id', 'lig8_id', 'lig9_id', 'lig10_id' ]
    pairs = pd.DataFrame(columns=cols)
    for i in predictions.iloc[:, 0].unique():
        best_pairs_i = list(predictions[predictions.idx_pro == i].nlargest(10, 'binding_proba').idx_lig)
        pairs = pairs.append(pd.DataFrame([[i]+best_pairs_i], columns=cols))
    return pairs


def main(model):
    """
    find the 10 best pairs according to the input model and save it to csv
    """
    model = keras.models.load_model(model)
    data_lig, data_pro = create_test_df()
    predictions = process_test_df(data_pro, data_lig, model)
    pairs = find_best_pairs(predictions)
    pairs.to_csv('best_pairs.csv')


if __name__ == "__main__":
    args = sys.argv
    main(args[1])


