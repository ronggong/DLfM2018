"""this script evaluate the pronunciation embedding,
embedding_classifier_ap evaluate the classification model,
"""

import csv
import os
import numpy as np
from eval_embedding_helper import ground_truth_matrix
from eval_embedding_helper import eval_embeddings_no_trim
from training_scripts.data_preparation import load_data_embedding_teacher_student
from src.parameters import config_select
from training_scripts.models_RNN import model_select
from training_scripts.models_RNN import model_select_attention
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from training_scripts.attention import Attention


def get_index_teacher_student(labels, label_integer_val):
    """Get teacher and student index"""
    phn_set = list(set([l.split('_')[0] for l in labels]))
    for ii in range(len(phn_set)):
        indices_phn = [i for i, s in enumerate(labels) if phn_set[ii] == s.split('_')[0]]
        label_integer_val[indices_phn] = ii
    index_teacher = [ii for ii in range(len(label_integer_val)) if 'teacher' in labels[ii]]
    index_student = [ii for ii in range(len(label_integer_val)) if 'student' in labels[ii]]
    return index_teacher, index_student


def calculate_ap(embeddings, label_integer_val, index_student):
    """calculate the distance matrix"""
    dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine'))) / 2.0
    gt_mat = ground_truth_matrix(label_integer_val)

    # we only compare teacher to student embeddings
    dist_mat = dist_mat[:min(index_student), min(index_student):]
    gt_mat = gt_mat[:min(index_student), min(index_student):]

    ap = eval_embeddings_no_trim(dist_mat=dist_mat, gt_mat=gt_mat)
    return ap


def run_eval(path_dataset, path_output, val_test, exp):

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

    # path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'

    if val_test == 'val':
        filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_val_teacher.pkl')
        filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_val_student.pkl')
        filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    elif val_test == 'test':
        filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_test_teacher.pkl')
        filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_test_extra_student.pkl')
        filename_list_key_student = os.path.join(path_dataset, 'list_key_extra_student.pkl')
    else:
        raise ValueError('val test is not valid.')

    filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')
    filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    if val_test == 'test':
        embedding_classifier_ap(filename_feature_teacher,
                                filename_list_key_teacher,
                                filename_feature_student,
                                filename_list_key_student,
                                filename_scaler,
                                config=[2, 0],
                                val_test='test',
                                attention=attention,
                                dense=dense,
                                conv=conv,
                                dropout=dropout,
                                path_eval=path_output)
    elif val_test == 'val':
        configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
        # configs = [[2, 0]]
        for config in configs:
            embedding_classifier_ap(filename_feature_teacher,
                                    filename_list_key_teacher,
                                    filename_feature_student,
                                    filename_list_key_student,
                                    filename_scaler,
                                    config=config,
                                    val_test='val',
                                    path_eval=path_output)


def embedding_classifier_ap(filename_feature_teacher,
                            filename_list_key_teacher,
                            filename_feature_student,
                            filename_list_key_student,
                            filename_scaler,
                            config,
                            val_test,
                            attention=False,
                            dense=False,
                            conv=False,
                            dropout=False,
                            path_eval="./eval/phone_embedding_classifier"):
    """calculate average precision of classification embedding"""

    list_feature_flatten_val, label_integer_val, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler)

    labels = le.inverse_transform(label_integer_val)

    index_teacher, index_student = get_index_teacher_student(labels=labels, label_integer_val=label_integer_val)

    path_model = './models/phone_embedding_classifier'
    # path_eval = './eval/phone_embedding_classifier'

    # for config in configs:
    model_name = config_select(config)

    list_ap = []
    embedding_dim = 27
    input_shape = [1, None, 80]

    prefix = '_27_class'

    if attention and conv and dropout:
        attention_dense_str = 'attention_conv_dropout_'
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

    # 5 running times
    for ii in range(5):

        # embedding model filename
        filename_model = os.path.join(path_model, model_name + prefix + '_' + attention_dense_str + str(ii) + '.h5')

        # load model weights, create a new model with batch size 1, then set weights back
        if attention:
            model = load_model(filepath=filename_model, custom_objects={'Attention': Attention(return_attention=True)})
            x, input, _ = model_select_attention(config=config, input_shape=input_shape, conv=conv, dropout=dropout)
        else:
            model = load_model(filepath=filename_model)
            x, input = model_select(config=config, input_shape=input_shape, conv=conv, dropout=dropout)

        weights = model.get_weights()

        if dense:
            outputs = Dense(units=32)(x)
        else:
            outputs = Dense(embedding_dim, activation='softmax')(x)

        model_1_batch = Model(inputs=input, outputs=outputs)

        model_1_batch.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

        print(model_1_batch.summary())

        model_1_batch.set_weights(weights=weights)

        embeddings = np.zeros((len(list_feature_flatten_val), embedding_dim))

        for ii_emb in range(len(list_feature_flatten_val)):

            print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_val), 'total')

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten_val[ii_emb]), axis=0)

            # evaluate the feature to get the embedding
            out = model_1_batch.predict_on_batch(x_batch)

            if attention:
                embeddings[ii_emb, :] = out[0, :]
            else:
                embeddings[ii_emb, :] = out

        ap = calculate_ap(embeddings=embeddings, label_integer_val=label_integer_val, index_student=index_student)

        list_ap.append(ap)

    post_fix = prefix if val_test == 'val' else prefix + '_extra_student'

    # write results to .csv
    filename_eval = os.path.join(path_eval, model_name + post_fix + '_' + attention_dense_str + '.csv')

    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="To evaluate the pronunciation embedding using validation or test set.")
    parser.add_argument("-d",
                        "--dataset_path",
                        type=str,
                        help="Type the dataset path")

    parser.add_argument("-o",
                        "--output_path",
                        type=str,
                        help="Type the results output path")

    parser.add_argument("-v",
                        "--valtest",
                        type=str,
                        default='test',
                        choices=['val', 'test'],
                        help="Choose validation or test.")

    parser.add_argument("-e",
                        "--experiment",
                        type=str,
                        default='baseline',
                        choices=['baseline', 'attention', 'dense', 'cnn', '32_embedding', 'dropout',
                                 'best_combination'],
                        help="choose the experiment.")

    args = parser.parse_args()

    run_eval(path_dataset=args.dataset_path,
             val_test=args.valtest,
             exp=args.experiment,
             path_output=args.output_path)
