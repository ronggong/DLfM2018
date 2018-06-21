import matplotlib
matplotlib.use('Tkagg')

import os
import numpy as np
import pickle
from training_scripts.data_preparation import load_data_embedding_all
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from training_scripts.models_RNN import model_select
from src.parameters import config_select


def embedding_classifier_helper(configs,
                                path_eval,
                                path_model,
                                le,
                                list_feature_flatten,
                                scaler,
                                label,
                                dense=False,
                                emb_all=False):
    """
    Output the classification model embedding for overall quality
    :param configs: model architecture
    :param path_eval: embedding output path
    :param path_model: embedding model path
    :param le: label encoder
    :param list_feature_flatten: log-mel feature list
    :param scaler: feature scaler
    :param label:
    :param dense: bool, use 32 embedding or not
    :param emb_all: bool, use all the dataset, including professional, amateur train, validation and extra test
    :return:
    """

    embedding_dim = 32 if dense else 2

    for config in configs:
        input_shape = [1, None, 80]
        prefix = '_2_class_teacher_student'
        dense_str = "dense_32_" if dense else ""
        model_name = config_select(config) + prefix

        if emb_all and dense:
            emb_all_str = "_dense_all"
        elif emb_all:
            emb_all_str = "_all"
        else:
            emb_all_str = ""

        if le:
            # label encoder
            pickle.dump(le, open(os.path.join(path_eval, model_name + '_le.pkl'), 'wb'), protocol=2)

        ii = 0  # only use the first model

        filename_model = os.path.join(path_model, model_name + '_' + dense_str + str(ii) + '.h5')

        model = load_model(filepath=filename_model)

        weights = model.get_weights()

        x, input = model_select(config=config, input_shape=input_shape, conv=False, dropout=False)

        if dense:
            outputs = Dense(embedding_dim)(x)
        else:
            outputs = Dense(embedding_dim, activation='softmax')(x)

        model_1_batch = Model(inputs=input, outputs=outputs)

        model_1_batch.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

        model_1_batch.set_weights(weights=weights)

        embeddings_profess = np.zeros((len(list_feature_flatten), embedding_dim))

        for ii_emb in range(len(list_feature_flatten)):
            print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten), 'total')

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten[ii_emb]), axis=0)

            embeddings_profess[ii_emb, :] = model_1_batch.predict_on_batch(x_batch)

        np.save(file=os.path.join(path_eval,
                                  model_name + '_embedding_overall_quality' + dense_str + emb_all_str + str(ii)),
                arr=embeddings_profess)

        np.save(file=os.path.join(path_eval,
                                  model_name + '_embeddings_labels' + dense_str + emb_all_str), arr=label)


def embedding_classifier_tsne_all(filename_feature_teacher_train,
                                  filename_feature_teacher_val,
                                  filename_feature_teacher_test,
                                  filename_list_key_teacher,
                                  filename_feature_student_train,
                                  filename_feature_student_val,
                                  filename_feature_student_test,
                                  filename_list_key_student,
                                  filename_feature_student_extra_test,
                                  filename_list_key_extra_student,
                                  filename_scaler,
                                  path_model="./models/phone_embedding_classifier",
                                  path_eval="./eval/phone_embedding_classifier",
                                  dense=False):

    list_feature_flatten, list_key_flatten, scaler = load_data_embedding_all(filename_feature_teacher_train,
                                                                             filename_feature_teacher_val,
                                                                             filename_feature_teacher_test,
                                                                             filename_list_key_teacher,
                                                                             filename_feature_student_train,
                                                                             filename_feature_student_val,
                                                                             filename_feature_student_test,
                                                                             filename_list_key_student,
                                                                             filename_feature_student_extra_test,
                                                                             filename_list_key_extra_student,
                                                                             filename_scaler)

    embedding_classifier_helper(configs=[[1, 0]],
                                path_eval=path_eval,
                                path_model=path_model,
                                le=False,
                                list_feature_flatten=list_feature_flatten,
                                scaler=scaler,
                                label=list_key_flatten,
                                emb_all=True,
                                dense=dense)


def run_tsne_embedding_extractor(data_path, output_path, model_path, dense):
    filename_feature_teacher_train = os.path.join(data_path, 'feature_phn_embedding_train_teacher.pkl')
    filename_feature_teacher_val = os.path.join(data_path, 'feature_phn_embedding_val_teacher.pkl')
    filename_feature_teacher_test = os.path.join(data_path, 'feature_phn_embedding_test_teacher.pkl')
    filename_list_key_teacher = os.path.join(data_path, 'list_key_teacher.pkl')

    filename_feature_student_train = os.path.join(data_path, 'feature_phn_embedding_train_student.pkl')
    filename_feature_student_val = os.path.join(data_path, 'feature_phn_embedding_val_student.pkl')
    filename_feature_student_test = os.path.join(data_path, 'feature_phn_embedding_test_student.pkl')
    filename_list_key_student = os.path.join(data_path, 'list_key_student.pkl')

    filename_feature_student_extra_test = os.path.join(data_path, 'feature_phn_embedding_test_extra_student.pkl')
    filename_list_key_extra_student = os.path.join(data_path, 'list_key_extra_student.pkl')

    filename_scaler = os.path.join(data_path, 'scaler_phn_embedding_train_teacher_student.pkl')

    embedding_classifier_tsne_all(filename_feature_teacher_train,
                                  filename_feature_teacher_val,
                                  filename_feature_teacher_test,
                                  filename_list_key_teacher,
                                  filename_feature_student_train,
                                  filename_feature_student_val,
                                  filename_feature_student_test,
                                  filename_list_key_student,
                                  filename_feature_student_extra_test,
                                  filename_list_key_extra_student,
                                  filename_scaler,
                                  path_model=model_path,
                                  path_eval=output_path,
                                  dense=dense)


if __name__ == '__main__':

    import argparse

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description="To extract the embeddings for t-SNE plotting.")
    parser.add_argument("-d",
                        "--dataset_path",
                        type=str,
                        help="Type the dataset path")

    parser.add_argument("-o",
                        "--output_path",
                        type=str,
                        help="Type the output path")

    parser.add_argument("-m",
                        "--model_path",
                        type=str,
                        help="Type the model path")

    parser.add_argument("--dense",
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default="False",
                        help="choose the dense.")

    args = parser.parse_args()

    run_tsne_embedding_extractor(data_path=args.dataset_path,
                                 output_path=args.output_path,
                                 model_path=args.model_path,
                                 dense=args.dense)

