import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Dropout, Flatten, BatchNormalization, Activation
import sklearn.metrics
import sys


def model(shape=(20, 20, 20, 8)):
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), activation='relu', input_shape=shape[1:], data_format="channels_last"))
    # model.add(MaxPooling3D((2, 2, 2)))
    # model.add(Conv3D(128, (5, 5, 5), activation='relu'))
    model.add(MaxPooling3D((2, 2, 2)))
    # model.add(Conv3D(256, (5, 5, 5), activation='relu'))
    # model.add(MaxPooling3D((2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    # model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train(train_x, train_y, test, batch_size, num_epochs):
    to_train = model(train_x.shape)
    to_train.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    to_train.fit(train_x, train_y, batch_size, epochs=num_epochs, validation_data=test)
    return to_train


def split_data(train_x, train_y, ratio, training_size):
    size = training_size
    permutation = np.random.permutation(len(train_y))
    test_indexes = permutation[:int(size * ratio)]
    train_indexes = permutation[int(size * ratio):size]
    test_x = train_x[test_indexes]
    test_y = train_y[test_indexes]
    train_x = train_x[train_indexes]
    train_y = train_y[train_indexes]
    return train_x[:training_size], train_y, (test_x, test_y)


def confusion_matrix(test_predictions, test_labels):
    rounded_test_predictions = [round(prediction[0]) for prediction in test_predictions]
    matrix = sklearn.metrics.confusion_matrix(test_labels, rounded_test_predictions)
    tn, fp, fn, tp = matrix.ravel()
    print("true positives : %s, true_negatives:%s, false_positives:%s, false_negatives:%s" % (tp, tn, fp,fn))
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    precision = tp/(tp+fp)
    print('accuracy: %s, precision: %s' %(accuracy, precision))


def main(batch_size=1, epochs=10, test_ratio=0.1, training_size=6000):
    print('args should be batch size, epochs, test_ratio, num training examples')
    train_x = np.load('training_data.npy')
    train_y = np.load('training_labels.npy')
    train_x, train_y, test = split_data(train_x, train_y, test_ratio, training_size)
    trained_model = train(train_x, train_y, test, batch_size, epochs)
    test_predictions = trained_model.predict(test[0], batch_size=1)
    filename = 'trained_model' + str(epochs) + '_epochs_' + str(batch_size)+'_batch.h5'
    confusion_matrix(test_predictions, test[1])
    trained_model.save(filename)


if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]), int(args[2]), float(args[3]), int(args[4]))




