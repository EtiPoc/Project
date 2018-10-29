import numpy as np
import pandas as pd
import keras
from reading import  Box
from testing_data_release.read_testing_pdb_file import read_pdb


def create_test_df(files_nb=824):
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
    # data_lig.drop('index', 1, inplace=True)
    data_pro.reset_index()
    # data_pro.drop('index', 1, inplace=True)
    return data_lig, data_pro


def process_test_df(data_pro, data_lig, model):
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


def predict(model, final_df):
    final_df['binding_proba'] = final_df.apply(lambda row: model.predict(row.box.reshape((1, 10, 10, 10, 8)))[0][0], 1)
    return final_df


def find_best_pairs(predictions):
    cols = ['pro_id', 'lig1_id', 'lig2_id', 'lig3_id', 'lig4_id', 'lig5_id', 'lig6_id', 'lig7_id', 'lig8_id', 'lig9_id', 'lig10_id' ]
    pairs = pd.DataFrame(columns=cols)
    for i in predictions.iloc[:, 0].unique():
        best_pairs_i = list(predictions[predictions.idx_pro == i].nlargest(10, 'binding_proba').idx_lig)
        pairs = pairs.append(pd.DataFrame([[i]+best_pairs_i], columns=cols))
    return pairs





# def main():
model = keras.models.load_model('94.17model-009.h5')
test_df = create_test_df()
predictions = process_test_df(test_df[0], test_df[1], model)
pairs = find_best_pairs(predictions)
pairs.to_csv('best_pairs.csv')



