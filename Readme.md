******CS5242 Deep Learning project******

E0338141 E0338139

The goal of this project is to develop a deep learning model that predicts whether a pair
protein/ligand will bind based on the coordinates of the atoms of each molecules.


**Prerequisites**

You need to have training data and testing_data_release at the root

Only basic modules such as keras or scikit-learn are required


**Running procedure**

python reading.py ratio
python model.py 1 10 0.1  6000
arguments are batch_size, num epochs, test_ratio, training_size (3000*(ratio+1))
python test.py model_name

Depending on the previous score or best epochs, models can have different names so you have to imput the one you want

By default, the voxel is 10 * 10 * 10 with 2A for each step. You can change that in reading.py