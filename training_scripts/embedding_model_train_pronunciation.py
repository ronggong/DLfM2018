"""
train the classification pronunciation embedding
"""

import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data_preparation import load_data_embedding_teacher_student
from keras.utils import to_categorical
from models_RNN import train_embedding_RNN_batch
from parameters import config_select


def run_process(path_dataset, path_model, exp):
    batch_size = 64
    input_shape = (batch_size, None, 80)
    patience = 15
    output_shape = 27

    attention = False
    conv = False
    dropout = False
    dense = False

    if exp == "baseline":
        pass
    elif exp == "attention":
        attention = True
    elif exp == "32_embedding":
        dense = True
    elif exp == "cnn":
        conv = True
    elif exp == "dense":
        dense = True
    elif exp == "dropout":
        dropout = 0.25
    elif exp == "best_combination":
        attention = True
        conv = True
        dropout = 0.25
    else:
        raise ValueError("exp {} is not a valid parameter".format(exp))

    if attention and conv and dropout:
        attention_dense_str = "attention_conv_dropout_"
    elif attention:
        attention_dense_str = "attention_"
    elif dense:
        attention_dense_str = "dense_"
    elif conv:
        attention_dense_str = "conv_"
    elif dropout:
        attention_dense_str = "dropout_"
    else:
        attention_dense_str = ""

    # path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'

    filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')
    filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    filename_scaler_teacher_student = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    filename_label_encoder = os.path.join(path_dataset, 'le_phn_embedding_teacher_student.pkl')
    filename_data_splits = os.path.join(path_dataset, 'data_splits_teacher_student.pkl')

    # path_model = '../../temp'

    list_feature_flatten, labels_integer, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler_teacher_student)

    # combine teacher and student label to the same one
    labels = le.inverse_transform(labels_integer)
    phn_set = list(set([l.split('_')[0] for l in labels]))
    for ii in range(len(phn_set)):
        indices_phn = [i for i, s in enumerate(labels) if phn_set[ii] == s.split('_')[0]]
        labels_integer[indices_phn] = ii

    train_index, val_index = pickle.load(open(filename_data_splits, 'rb'))

    # for train_index, val_index in folds5_split_indices:
    if attention or dense or conv or dropout:
        configs = [[2, 0]]
    else:
        configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]

    for config in configs:

        model_name = config_select(config=config)

        for ii in range(5):
            file_path_model = os.path.join(path_model,
                                           model_name + '_27_class' + '_' + attention_dense_str + str(ii) + '.h5')
            file_path_log = os.path.join(path_model, 'log',
                                         model_name + '_27_class' + '_' + attention_dense_str + str(ii) + '.csv')

            list_feature_fold_train = [scaler.transform(list_feature_flatten[ii]) for ii in train_index]
            labels_integer_fold_train = labels_integer[train_index]
            labels_fold_train = to_categorical(labels_integer_fold_train)

            list_feature_fold_val = [scaler.transform(list_feature_flatten[ii]) for ii in val_index]
            labels_integer_fold_val = labels_integer[val_index]
            labels_fold_val = to_categorical(labels_integer_fold_val)

            train_embedding_RNN_batch(list_feature_fold_train=list_feature_fold_train,
                                      labels_fold_train=labels_fold_train,
                                      list_feature_fold_val=list_feature_fold_val,
                                      labels_fold_val=labels_fold_val,
                                      batch_size=batch_size,
                                      input_shape=input_shape,
                                      output_shape=output_shape,
                                      file_path_model=file_path_model,
                                      filename_log=file_path_log,
                                      patience=patience,
                                      config=config,
                                      attention=attention,
                                      dense=dense,
                                      conv=conv,
                                      dropout=dropout)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="To reproduce the experiment results of pronunciation embedding.")
    parser.add_argument("-d",
                        "--dataset_path",
                        type=str,
                        help="Type the dataset path")

    parser.add_argument("-o",
                        "--output_path",
                        type=str,
                        help="Type the output path")

    parser.add_argument("-e",
                        "--experiment",
                        type=str,
                        default='baseline',
                        choices=['baseline', 'attention', 'dense', 'cnn', '32_embedding', 'dropout',
                                 'best_combination'],
                        help="choose the experiment.")

    args = parser.parse_args()

    run_process(path_dataset=args.dataset_path,
                path_model=args.output_path,
                exp=args.experiment)
