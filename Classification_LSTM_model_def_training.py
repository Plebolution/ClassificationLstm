import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from sklearn.model_selection import train_test_split



def getting_data(log_path, labels_ids_path, keys_path):
    
    """ Function for reading files

        Reading files and forming lists of data.

        Args:
            log_path: directory of keys workflow file.
            labels_ids_path: directory of keys ids workflow file.
            keys_path: directory of list of keys file.

        Returns:
            3 lists of data, formed from read files.

        Raises:
            FileNotFoundError: file or directory does not exist.
            IsADirectoryError: expected file, but its a directory.
            PermissionError: not enough access rights.

    """
    
    with open(log_path) as log_file:
        log_list = [line.strip('\n') for line in log_file]
        log_list = [line for line in log_list if line]

    with open(labels_ids_path) as label_ids_file:
        label_ids_list = [line.strip('\n') for line in label_ids_file]
        label_ids_list = [line for line in label_ids_list if line]
    
    with open(keys_path) as keys_file:
        keys_list = [line.strip('\n') for line in keys_file]
        keys_list = [line for line in keys_list if line]

    return log_list, label_ids_list, keys_list



def data_preprocessing(h, train_log_list, train_label_ids_list,
                        train_keys_list):

    """ Preprocessing of input data
    
        Creating dictionary of keys and its int values, forming 1 dim lists of
        workflow with int values of keys. Vectorization of formed 1 dim lists
        and creating 3 dim lists for neural network input.

        Args:
            h: length of token sequence for training and prediction.
            train_log_list: list of strings of training workflow of keys.
            traing_label_ids_list: list strings of training keys ids workflow.
            train_keys_list: list of strings of training keys.
            dataX: 1 dim list of int values of keys workflow.
            dataY: 1 dim list of keys ids workflow.
            labels_dict: dictionary, which converts string keys to ints.
            x_train: 3 dim list of vectorized keys workflow.
            y_train: one-hot encoded keys ids workflow.

        Returns:
            x_train, y_train and count of keys.
    
    """

    dataX = []
    dataY = []

    input_vocab_size = len(train_keys_list)

    classes_count = len(train_keys_list)
 
    labels_dict = dict(zip(train_keys_list, 
        [i for i in range(0, len(train_keys_list))]))

    for i in range(0, len(train_log_list) - h):
        temp = train_log_list[i: i + h]
        dataX.append([labels_dict[item] for item in temp])
        dataY.append(train_label_ids_list[i + h])

    x_train = np.reshape(dataX,(len(dataX), h, 1))
    x_train = x_train / float(input_vocab_size)
    y_train = np_utils.to_categorical(dataY)

    return x_train, y_train, classes_count, input_vocab_size



def model_def_train_test(epochs, batch_size, x_train, y_train, x_test, y_test,
                        classes_count, input_vocab_size):

    """ Model definition

        Sequential model definition with 2 LSTM layers, and output Dense layer.
        Checkpointing and saving weights improvement while fitting model. Also
        saving model information (shapes, batch_size, classes_count). Taking
        test data sequence and testing model. Training data is 2/3 of all data,
        testing data is last 1/3 of all data.

        Args:
            epochs: int count of model training stage iterations.
            batch_size: int model batch size.
            x_train: numpy.ndarray 3 dim list of vectorized training keys 
              workflow.
            y_train: numpy.ndarray one-hot encoded keys ids training workflow.
            x_test: a part of training set of x for testing.
            y_test: a part of training set of y for testing.
            classes_count: count of training keys.
            input_vocab_size: count of training keys.

    """

    model_info_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//model_info.txt'
    model_save_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//model_config.h5'
    weights_path="C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//model_weights//weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

    model = Sequential()
    model.add(LSTM(64,
        input_shape=(x_train.shape[1], x_train.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(classes_count, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    print(model.summary())

    checkpoint = ModelCheckpoint(weights_path, monitor='loss', verbose=1,
        save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    model.fit(x_train, y_train,
        epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    
    model.save(model_save_path)

    with open(model_info_path, 'w') as path:
        path.write(str(input_vocab_size) + '\n')
        path.write(str(classes_count) + '\n')
        path.write(str(x_train.shape[1]) + '\n')
        path.write(str(x_train.shape[2]))

    accr = model.evaluate(x_test, y_test)
    print('Test set\nLoss: {:0.3f}\nAccuracy: {:0.3f}'.format(accr[0],accr[1]))



def main():

    """ 
        Setting neural network parameters and needed files paths. Splitting
        data on training set and testing set.

        Args:
            log_train_path: string directory of training keys workflow file.
            label_ids_train_path: string directory of training keys ids
              workflow file.
            keys_train_path: string directory of training list of keys file.
            train_log_list: list of strings of training workflow of keys.
            traing_label_ids_list: list strings of training keys ids workflow.
            train_keys_list: list of strings of training keys.
            h: int length of token sequence for training and prediction.
            epochs: int count of model training stage iterations.
            batch_size: int model batch size.
            x_train: numpy.ndarray 3 dim list of vectorized training keys 
              workflow.
            y_train: numpy.ndarray one-hot encoded keys ids training workflow.
            x_test: a part of training set of x for testing.
            y_test: a part of training set of y for testing.
            clases_count: count of training keys.
    
    """

    h = 3
    epochs = 3
    batch_size = 1

    log_train_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//Workflow.txt'
    label_ids_train_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//WorkflowID.txt'
    keys_train_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//LogKeys.txt'

    train_log_list, train_label_ids_list, train_keys_list = getting_data(
        log_train_path, label_ids_train_path, keys_train_path)    
    
    x_train, y_train, classes_count, input_vocab_size = data_preprocessing(
        h, train_log_list, train_label_ids_list, train_keys_list)
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
    
    model_def_train_test(epochs, batch_size, x_train, y_train, x_test, y_test,
        classes_count, input_vocab_size)



if __name__ == '__main__':
    main()