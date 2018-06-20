import csv
import os
import numpy as np
import pandas as pd
from eval_embedding_helper import ground_truth_matrix
from training_scripts.data_preparation import load_data_embedding_teacher_student
from src.parameters import config_select
from training_scripts.models_RNN import model_select
from training_scripts.models_RNN import model_select_attention
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score
from training_scripts.attention import Attention


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
        dense = True
        conv = True
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
                                config=[1, 0],
                                val_test=val_test,
                                attention=attention,
                                dense=dense,
                                conv=conv,
                                dropout=dropout,
                                path_eval=path_output)
    else:
        configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
        for config in configs:
            embedding_classifier_ap(filename_feature_teacher,
                                    filename_list_key_teacher,
                                    filename_feature_student,
                                    filename_list_key_student,
                                    filename_scaler,
                                    config=config,
                                    val_test=val_test,
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
                            path_eval='./eval/phone_embedding_classifier'):
    """calculate teacher student pairs average precision of classifier embedding"""

    list_feature_flatten_val, label_integer_val, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler)

    path_model = './models/phone_embedding_classifier'
    # path_eval = './eval/phone_embedding_classifier'

    prefix = '_2_class_teacher_student'

    if dense:
        embedding_dim = 32
    else:
        embedding_dim = 2

    model_name = config_select(config) + prefix

    if dense and conv:
        attention_dense_str = 'dense_conv_'
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

    list_ap = []
    input_shape = [1, None, 80]

    # average precision of each phone
    array_ap_phn_5_runs = np.zeros((5, 27))

    for ii in range(5):
        print('run time', ii)
        filename_model = os.path.join(path_model, model_name + '_' + attention_dense_str + str(ii) + '.h5')
        if attention:
            model = load_model(filepath=filename_model, custom_objects={'Attention': Attention(return_attention=True)})
            x, input, _ = model_select_attention(config=config, input_shape=input_shape, conv=conv, dropout=dropout)
        else:
            model = load_model(filepath=filename_model)
            x, input = model_select(config=config, input_shape=input_shape, conv=conv, dropout=dropout)

        weights = model.get_weights()

        if dense:
            outputs = Dense(embedding_dim)(x)
        else:
            outputs = Dense(embedding_dim, activation='softmax')(x)

        model_1_batch = Model(inputs=input, outputs=outputs)

        model_1_batch.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
        model_1_batch.set_weights(weights=weights)

        embeddings = np.zeros((len(list_feature_flatten_val), embedding_dim))

        for ii_emb in range(len(list_feature_flatten_val)):

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten_val[ii_emb]), axis=0)

            out = model_1_batch.predict_on_batch(x_batch)

            if attention:
                embeddings[ii_emb, :] = out[0, :]
            else:
                embeddings[ii_emb, :] = out

        list_dist = []
        list_gt = []
        array_ap_phn = np.zeros((27,))
        cols = []
        list_ratio_tea_stu = []
        # calculate the AP for each phone
        for ii_class in range(27):
            # teacher student pair class index
            idx_ii_class = np.where(np.logical_or(label_integer_val == 2*ii_class,
                                                  label_integer_val == 2*ii_class+1))[0]

            idx_ii_class_stu = len(np.where(label_integer_val == 2*ii_class)[0])
            idx_ii_class_tea = len(np.where(label_integer_val == 2*ii_class+1)[0])

            # ratio of teacher's samples
            list_ratio_tea_stu.append(idx_ii_class_tea/float(idx_ii_class_tea+idx_ii_class_stu))

            dist_mat = (2.0 - squareform(pdist(embeddings[idx_ii_class], 'cosine')))/2.0
            labels_ii_class = [label_integer_val[idx] for idx in idx_ii_class]
            gt_mat = ground_truth_matrix(labels_ii_class)

            sample_num = dist_mat.shape[0]
            iu1 = np.triu_indices(sample_num, 1)  # trim the upper mat

            list_dist.append(dist_mat[iu1])
            list_gt.append(gt_mat[iu1])

            # calculate the average precision of each phoneme
            ap_phn = average_precision_score(y_true=np.abs(list_gt[ii_class]),
                                             y_score=np.abs(list_dist[ii_class]),
                                             average='weighted')

            cols.append(le.inverse_transform(2*ii_class).split('_')[0])
            array_ap_phn[ii_class] = ap_phn

        array_dist = np.concatenate(list_dist)
        array_gt = np.concatenate(list_gt)

        ap = average_precision_score(y_true=np.abs(array_gt), y_score=np.abs(array_dist), average='weighted')

        list_ap.append(ap)

        array_ap_phn_5_runs[ii, :] = array_ap_phn

    # save results to .csv
    post_fix = prefix+'_2_class' if val_test == 'val' else prefix+'_2_class_extra_student'

    filename_eval = os.path.join(path_eval, model_name + post_fix + attention_dense_str + '.csv')

    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])

    # organize the Dataframe, and save the individual phone results
    ap_phn_mean = np.mean(array_ap_phn_5_runs, axis=0)
    ap_phn_std = np.std(array_ap_phn_5_runs, axis=0)
    ap_phn_mean_std = pd.DataFrame(np.transpose(np.vstack((ap_phn_mean, ap_phn_std, list_ratio_tea_stu))),
                                   columns=['mean', 'std', 'ratio'],
                                   index=cols)

    ap_phn_mean_std = ap_phn_mean_std.sort_values(by='mean')
    ap_phn_mean_std.to_csv(os.path.join(path_eval,
                                        model_name + post_fix + attention_dense_str + '_phn_mean_std.csv'))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="To evaluate the overall quality embedding using validation or test set.")
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
