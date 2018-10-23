import pandas as pd


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
            atomtype_list.append('h')  # 'h' means hydrophobic
        else:
            atomtype_list.append('p')  # 'p' means polar

    return X_list, Y_list, Z_list, atomtype_list


# X_list, Y_list, Z_list, atomtype_list = read_pdb(
#     "training_data/2060_lig_cg.pdb")
# X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_pro_cg.pdb")
# X_list, Y_list, Z_list, atomtype_list=read_pdb("training_first_100_samples/0001_lig_cg.pdb")
# print(X_list)
# print(Y_list)
# print(Z_list)
# print(atomtype_list)


lig = pd.DataFrame()
for i in range(1, 3001):
    number = (4-len(str(i)))*"0" + str(i)
    filename = "training_data/"+number+"_lig_cg.pdb"
    X_list, Y_list, Z_list, atomtype_list = read_pdb(filename)
    lig = lig.append(pd.DataFrame([[i, X_list, Y_list, Z_list, atomtype_list]]))
lig.set_index(0, inplace=True)
lig.columns = ['X_list', 'Y_list', 'Z_list', 'atomtype_list']
lig.to_csv('training_data_lig.csv')
