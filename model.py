import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
import sklearn.metrics
import sys
import matplotlib.pyplot as plt


def model(shape=(20, 20, 20, 8)):
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3), activation='relu', input_shape=shape[1:], data_format="channels_last", padding="same"))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding="same"))
    # model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(512, (3, 3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model


def train(train_x, train_y, test, batch_size, num_epochs):
    to_train = model(train_x.shape)
    #checkpoint that allows to save the best model during the training
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=0, monitor='val_acc', save_best_only=True,
                                mode='auto')

    to_train.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    to_train.fit(train_x, train_y, batch_size, epochs=num_epochs, validation_data=test, callbacks=[checkpoint])
    return to_train


def split_data(train_x, train_y, ratio, training_size):
    """randomly select training_size sample in the dataset
     split the training set into train and test with the given ratio"""
    size = training_size
    permutation = np.random.permutation(len(train_y))
    test_indexes = permutation[:int(size * ratio)]
    train_indexes = permutation[int(size * ratio):size]
    test_x = train_x[test_indexes]
    test_y = train_y[test_indexes]
    train_x = train_x[train_indexes]
    train_y = train_y[train_indexes]
    return train_x, train_y, (test_x, test_y)


def confusion_matrix(test_predictions, test_labels):
    """compute the confusion matrix with the given predictions and true labels"""
    rounded_test_predictions = [round(prediction[0]) for prediction in test_predictions]
    matrix = sklearn.metrics.confusion_matrix(test_labels, rounded_test_predictions)
    tn, fp, fn, tp = matrix.ravel()
    print("true positives : %s, true_negatives:%s, false_positives:%s, false_negatives:%s" % (tp, tn, fp,fn))
    accuracy = (tp+tn)/(tp+tn+fn+fp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print('accuracy: %s, precision: %s, recall: %s' %(accuracy, precision, recall))
    return accuracy, precision, recall


def save_plots(trained_model):
    history = trained_model.history
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model_acc_plot_.png')
    plt.close()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model_loss_plot_.png')
    plt.close()


def main(batch_size=1, epochs=10, test_ratio=0.1, training_size=6000):
    print('args should be batch size, epochs, test_ratio, num training examples')

    train_x = np.load('training_data.npy')
    train_y = np.load('training_labels.npy')

    train_x, train_y, test = split_data(train_x, train_y, test_ratio, training_size)

    trained_model = train(train_x, train_y, test, batch_size, epochs)

    filename = 'trained_model_'+str(trained_model.history.history['val_acc'][-1])+'.h5'
    np.save('test_set_'+'trained_model_'+str(trained_model.history.history['val_acc'][-1])+'.npy', test)

    trained_model.save(filename)
    save_plots(trained_model)

    test_predictions = trained_model.predict(test[0], batch_size=1)
    confusion_matrix(test_predictions, test[1])



if __name__ == "__main__":
    args = sys.argv
    main(int(args[1]), int(args[2]), float(args[3]), int(args[4]))
