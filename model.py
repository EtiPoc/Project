import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Dropout, Flatten
import argparse
import sys


def model(shape=(20, 20, 20, 8)):
    model = Sequential()
    model.add(Conv3D(64, (5, 5, 5), activation='relu', input_shape=shape, data_format="channels_last"))
    # model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(128, (5, 5, 5), activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(256, (5, 5, 5), activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train(train_x, train_y, test, batch_size, num_epochs):
    to_train = model()
    to_train.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    to_train.fit(train_x, train_y, batch_size, epochs=num_epochs, validation_data=test)
    return to_train


def split_data(train_x, train_y, ratio):
    size = len(train_y)
    permutation = np.random.permutation(size)
    test_indexes = permutation[:int(size * ratio)]
    test_x = train_x[test_indexes]
    test_y = train_y[test_indexes]
    train_x = np.array([train_x[i] for i in range(size) if i not in test_indexes])
    train_y = np.array([train_y[i] for i in range(size) if i not in test_indexes])
    return train_x, train_y, (test_x, test_y)


def main(batch_size=1, epochs=10, test_ratio=0.1, training_size= 6000):
    print('args should be batch size, epochs, test_ratio, num training examples')
    train_x = np.load('training_data.npy')
    train_y = np.load('training_labels.npy')
    train_x, train_y, test = split_data(train_x[:training_size], train_y[:training_size], test_ratio)
    trained_model = train(train_x, train_y, test, batch_size, epochs)
    filename = 'trained_model' + str(epochs) + '_epochs_' + str(batch_size)+'_batch.h5'
    trained_model.save(filename)


if __name__ == "__main__":
    args = sys.argv
    print(int(args[1]), int(args[2]), float(args[3]))
    main(int(args[1]), int(args[2]), float(args[3]), int(args[4]))




