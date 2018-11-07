import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Activation

from tkinter import messagebox



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



def prediction_and_comparison(model, sequence, labels_ids,
                                labels, h, l, input_vocab_size):

    """ Prediction function

        Creating dictionaries for converting keys from string to int and back.
        Chosing seed for prediction. New token prediction in loop for l,
        comparison of predicted value and real, asking user "if this anomaly?".
        If it is not anomaly, adding this case to labels_accordance_dict
        dictionary. After all doing shift on by one element by adding to seed
        predicted value and popping out first element.

        Args:
            model: all of model information, such as layers, batch_size, nodes.
            sequence: string sequence needed for seed for prediction.
            labels_ids: list of strings of keys ids workflow for prediction.
            labels: list of strings of keys for prediction.
            l: int length of predicted sequence.
            h: int length of token sequence for training and prediction.
            input_vocab_size: prediction keys count.
            labels_dict: dictionary of keys, for converting string keys to int.
            inverted_labels_dict: dictionary for inverse keys converting.
            labels_accordance_dict: dictionary for accordance between predicted
            and real values of workflow keys.
            seed: string seed for prediction.
            res_seq: result string sequence.
        
        Return:
            List of string, which contains started seed and subsequent
            predicted values.
        
    """

    labels_dict = dict(zip(labels, [i for i in range(0, len(labels))]))
    inverted_labels_dict = {v: k for k, v in labels_dict.items()}
    labels_accordance_dict = dict(zip(labels, [[] for i in labels]))

    sequence = [labels_dict[item] for item in sequence]

    seed = sequence[0: h]
    sequence = [inverted_labels_dict[item] for item in sequence]

    res_seq = [] * len(sequence) * l
    res_seq.extend([inverted_labels_dict[item] for item in seed])
    
    for i in range(0, l):
        seq = np.reshape(seed, (1, len(seed), 1))
        seq = seq / float(input_vocab_size)
        pr = model.predict(seq, verbose=0)
        index = np.argmax(pr)
        pred = inverted_labels_dict[index]
        if sequence[h + i] != pred:
            if pred not in labels_accordance_dict[sequence[h + i]]:
                user_answer = messagebox.askyesno("Warning!", f"Anomaly detected, is it okay?\n {sequence[h + i]} == {pred}")
                if user_answer:
                    labels_accordance_dict[sequence[h + i]].append(pred)
                else:
                    messagebox.showerror("Warning!", "Need to fix!!!")
        res_seq.append(pred)
        seed.append(labels_dict[pred])
        seed.pop(0)
    
    return res_seq



def main():
    
    """ 
        Getting prediction data and calling function for predict.

        Args:
            l: int length of predicted sequence.
            h: int length of token sequence for training and prediction.
            input_vocab_size: count of training keys.
            pred_log_list: list of strings of workflow of keys for prediction.
            pred_label_ids_list: list of strings of keys ids workflow
              for prediction.
            pred_keys_list: list of strings of keys for prediction.
            pred_seq: predicted sequence string list.

    """

    l = 15

    log_predict_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//Workflow.txt' 
    label_ids_predict_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//WorkflowID.txt'
    keys_predict_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//LogKeys.txt'

    model = load_model('C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//model_config.h5')

    model_info_path = 'C://Users//vgolubch//Desktop//LSTMtest//LSTM_deeplog//model_info.txt'
    

    with open(model_info_path) as path:
        model_info_list = [line.strip('\n') for line in path]

    input_vocab_size = int(model_info_list[0])
    h = int(model_info_list[2])

    pred_log_list, pred_label_ids_list, pred_keys_list = getting_data(
        log_predict_path, label_ids_predict_path, keys_predict_path)

    pred_seq = prediction_and_comparison(model, pred_log_list,
        pred_label_ids_list, pred_keys_list, h, l, input_vocab_size)

    print('\n'.join(pred_seq))



if __name__ == "__main__":
    main()