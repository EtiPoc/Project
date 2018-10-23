import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Dropout, Flatten


def model(shape=(20, 20, 20, 8)):
    model = Sequential()
    model.add(Conv3D(64, (5, 5, 5), activation='relu', input_shape=shape, data_format="channels_last"))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(128, (5, 5, 5), activation='relu', data_format="channels_last"))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(256, (5, 5, 5), activation='relu', data_format="channels_last"))
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


def train(train_x, train_y):
    to_train = model()
    to_train.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    to_train.fit(train_x, train_y, 5, epochs=10, validation_split=0.1)

