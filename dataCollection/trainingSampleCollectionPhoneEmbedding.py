import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pickle
import soundfile as sf
from src.filePath import *
from src.parameters import *
from src.phonemeMap import dic_pho_map
from src.phonemeMap import dic_pho_label
from src.textgridParser import syllableTextgridExtraction
from src.audio_preprocessing import getMFCCBandsMadmom
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from src.train_test_filenames import getTeacherRecordings
from src.train_test_filenames import getStudentRecordings
from src.train_test_filenames import getExtraStudentRecordings


def dumpFeaturePho(wav_path,
                   textgrid_path,
                   recordings,
                   syllableTierName,
                   phonemeTierName):
    """
    dump the log-mel feature for each phoneme
    :param wav_path: path of the wavs
    :param textgrid_path: path of the textgrid
    :param recordings: recording file names
    :param syllableTierName: syllable textgrid tier name - dianSilence
    :param phonemeTierName: phoneme textgrid tier name - details
    :return:
    """

    # feature dictionary
    dic_pho_embedding = {}

    for _, pho in enumerate(set(dic_pho_map.values())):
        dic_pho_embedding[pho] = []

    for artist_path, recording in recordings:
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,
                                         join(artist_path, recording),
                                         syllableTierName,
                                         phonemeTierName)

        wav_full_filename = join(wav_path, artist_path, recording+'.wav')

        log_mel = getMFCCBandsMadmom(audio_fn=wav_full_filename, fs=fs, hopsize_t=hopsize_t)

        # go through all the phonemes in the list
        for ii, pho in enumerate(nestedPhonemeLists):
            print('calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists)))
            for p in pho[1]:
                # map from annotated xsampa to a readable notation with a dictionary
                try:
                    key = dic_pho_map[p[2]]
                except KeyError:
                    print(artist_path, recording)
                    print(ii, p[2])
                    raise

                sf = int(round(p[0] * fs / float(hopsize)))  # starting frame
                ef = int(round(p[1] * fs / float(hopsize)))  # ending frame

                log_mel_p = log_mel[sf:ef, :]  # log-mel phoneme

                if len(log_mel_p):
                    dic_pho_embedding[key].append(log_mel_p)

    return dic_pho_embedding


def dumpAudioPhn(wav_path,
                 textgrid_path,
                 recordings,
                 lineTierName,
                 phonemeTierName):
    """
    Dump audio of each phoneme
    :param wav_path:
    :param textgrid_path:
    :param recordings:
    :param lineTierName:
    :param phonemeTierName:
    :return:
    """

    ##-- dictionary feature
    dic_pho_wav = {}

    for _, pho in enumerate(set(dic_pho_map.values())):
        dic_pho_wav[pho] = []

    for artist_path, recording in recordings:
        nestedPhonemeLists, numSyllables, numPhonemes \
            = syllableTextgridExtraction(textgrid_path,
                                         join(artist_path, recording),
                                         lineTierName,
                                         phonemeTierName)

        # audio
        wav_full_filename = join(wav_path, artist_path, recording + '.wav')

        data_wav, fs_wav = sf.read(wav_full_filename)

        for ii, pho in enumerate(nestedPhonemeLists):
            print('calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists)))
            for p in pho[1]:
                # map from annotated xsampa to readable notation
                try:
                    key = dic_pho_map[p[2]]
                except KeyError:
                    print(artist_path, recording)
                    print(ii, p[2])
                    raise

                st = int(round(p[0] * fs_wav))  # starting time
                et = int(round(p[1] * fs_wav))  # ending time

                pho_wav = data_wav[st: et]

                if len(pho_wav):
                    dic_pho_wav[key].append(pho_wav)

    return dic_pho_wav


def featureAggregator(dic_pho_feature_train):
    """
    Aggregate feature dictionary into numpy array and label lists, and reshape the feature
    :param dic_pho_feature_train:
    :return:
    """
    feature_all = np.array([], dtype='float32')
    label_all = []
    for key in dic_pho_feature_train:
        feature = dic_pho_feature_train[key]
        label = [dic_pho_label[key]] * len(feature)

        if len(feature):
            if not len(feature_all):
                feature_all = feature
            else:
                feature_all = np.vstack((feature_all, feature))
            label_all += label

    label_all = np.array(label_all, dtype='int64')

    scaler = preprocessing.StandardScaler().fit(feature_all)

    return feature_all, label_all, scaler


def getTeacherStudentRecordings():
    """get teacher and student phoneme log mel features"""

    trainNacta2017_teacher, trainNacta_teacher, trainSepa_teacher, trainPrimarySchool_teacher = getTeacherRecordings()
    valPrimarySchool_student, trainPrimarySchool_student = getStudentRecordings()

    dic_pho_embedding_nacta2017_teacher = dumpFeaturePho(wav_path=nacta2017_wav_path,
                                                         textgrid_path=nacta2017_textgrid_path,
                                                         recordings=trainNacta2017_teacher,
                                                         syllableTierName='line',
                                                         phonemeTierName='details')

    dic_pho_embedding_nacta_teacher = dumpFeaturePho(wav_path=nacta_wav_path,
                                                     textgrid_path=nacta_textgrid_path,
                                                     recordings=trainNacta_teacher,
                                                     syllableTierName='line',
                                                     phonemeTierName='details')

    dic_pho_embedding_primarySchool_teacher = dumpFeaturePho(wav_path=primarySchool_wav_path,
                                                             textgrid_path=primarySchool_textgrid_path,
                                                             recordings=trainPrimarySchool_teacher,
                                                             syllableTierName='line',
                                                             phonemeTierName='details')

    dic_pho_embedding_sepa_teacher = dumpFeaturePho(wav_path=nacta_wav_path,
                                                    textgrid_path=nacta_textgrid_path,
                                                    recordings=trainSepa_teacher,
                                                    syllableTierName='line',
                                                    phonemeTierName='details')

    dic_pho_embedding_primarySchool_student = dumpFeaturePho(wav_path=primarySchool_wav_path,
                                                             textgrid_path=primarySchool_textgrid_path,
                                                             recordings=valPrimarySchool_student+trainPrimarySchool_student,
                                                             syllableTierName='line',
                                                             phonemeTierName='details')

    # fuse two dictionaries
    list_key_teacher = list(set(list(dic_pho_embedding_nacta_teacher.keys()) + list(dic_pho_embedding_nacta2017_teacher.keys()) +
                        list(dic_pho_embedding_primarySchool_teacher.keys()) + list(dic_pho_embedding_sepa_teacher.keys())))
    list_key_student = list(set(list(dic_pho_embedding_primarySchool_student.keys())))

    print(list_key_teacher)
    print(list_key_student)

    # teacher's part
    dic_pho_embedding_all = {}
    dic_pho_embedding_train = {}
    dic_pho_embedding_val = {}
    dic_pho_embedding_test = {}

    dic_pho_feature_train = {}

    feature_phn_train = []
    feature_phn_val = []
    feature_phn_test = []
    for key in list_key_teacher:
        dic_pho_embedding_all[key] = dic_pho_embedding_nacta2017_teacher[key] + dic_pho_embedding_nacta_teacher[key] + \
                                     dic_pho_embedding_primarySchool_teacher[key] + dic_pho_embedding_sepa_teacher[key]

        # split 20 percent for test
        dic_pho_embedding_train[key], dic_pho_embedding_test[key] = \
            train_test_split(dic_pho_embedding_all[key], test_size=0.2)

        dic_pho_embedding_train[key], dic_pho_embedding_val[key] = \
            train_test_split(dic_pho_embedding_train[key], test_size=0.2)

        feature_phn_train.append(dic_pho_embedding_train[key])
        feature_phn_val.append(dic_pho_embedding_val[key])
        feature_phn_test.append(dic_pho_embedding_test[key])
        dic_pho_feature_train[key] = np.vstack(dic_pho_embedding_train[key])

    # save feature label scaler
    # training and test data on the phone level
    filename_pho_embedding_train = join(data_path_phone_embedding_model, 'feature_phn_embedding_train_teacher.pkl')
    pickle.dump(feature_phn_train, open(filename_pho_embedding_train, 'wb'), protocol=2)

    filename_pho_embedding_val = join(data_path_phone_embedding_model, 'feature_phn_embedding_val_teacher.pkl')
    pickle.dump(feature_phn_val, open(filename_pho_embedding_val, 'wb'), protocol=2)

    filename_pho_embedding_test = join(data_path_phone_embedding_model, 'feature_phn_embedding_test_teacher.pkl')
    pickle.dump(feature_phn_test, open(filename_pho_embedding_test, 'wb'), protocol=2)

    pickle.dump(list_key_teacher,
                open(join(data_path_phone_embedding_model, 'list_key_teacher.pkl'), 'wb'), protocol=2)

    # student part
    dic_pho_embedding_train = {}
    dic_pho_embedding_val = {}
    dic_pho_embedding_test = {}

    feature_phn_train = []
    feature_phn_val = []
    feature_phn_test = []
    for key in list_key_student:
        # split 20 percent for test
        dic_pho_embedding_train[key], dic_pho_embedding_test[key] = \
            train_test_split(dic_pho_embedding_primarySchool_student[key], test_size=0.2)

        dic_pho_embedding_train[key], dic_pho_embedding_val[key] = \
            train_test_split(dic_pho_embedding_train[key], test_size=0.2)

        feature_phn_train.append(dic_pho_embedding_train[key])
        feature_phn_val.append(dic_pho_embedding_val[key])
        feature_phn_test.append(dic_pho_embedding_test[key])

        dic_pho_feature_train[key] = np.vstack(dic_pho_embedding_train[key])

    feature_train, _, scaler = featureAggregator(dic_pho_feature_train)

    # save feature label scaler
    # training and test data on the phone level, no need to save teacher's data
    filename_pho_embedding_train = join(data_path_phone_embedding_model, 'feature_phn_embedding_train_student.pkl')
    pickle.dump(feature_phn_train, open(filename_pho_embedding_train, 'wb'), protocol=2)

    filename_pho_embedding_val = join(data_path_phone_embedding_model, 'feature_phn_embedding_val_student.pkl')
    pickle.dump(feature_phn_val, open(filename_pho_embedding_val, 'wb'), protocol=2)

    filename_pho_embedding_test = join(data_path_phone_embedding_model, 'feature_phn_embedding_test_student.pkl')
    pickle.dump(feature_phn_test, open(filename_pho_embedding_test, 'wb'), protocol=2)

    pickle.dump(scaler,
                open(join(data_path_phone_embedding_model, 'scaler_phn_embedding_train_teacher_student.pkl'), 'wb'), protocol=2)

    pickle.dump(list_key_student,
                open(join(data_path_phone_embedding_model, 'list_key_student.pkl'), 'wb'), protocol=2)


def getExtraTestRecordings():
    """get extra phoneme log mel features"""

    extra_test_adult = getExtraStudentRecordings()

    dic_pho_embedding_extra_adult = dumpFeaturePho(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   recordings=extra_test_adult,
                                                   syllableTierName='line',
                                                   phonemeTierName='details')

    # fuse two dictionaries
    list_key_student = list(set(list(dic_pho_embedding_extra_adult.keys())))

    print(list_key_student)

    # student part
    feature_phn_test = []
    for key in list_key_student:
        feature_phn_test.append(dic_pho_embedding_extra_adult[key])

    # save feature
    filename_pho_embedding_test = join(data_path_phone_embedding_model, 'feature_phn_embedding_test_extra_student.pkl')
    pickle.dump(feature_phn_test, open(filename_pho_embedding_test, 'wb'), protocol=2)

    pickle.dump(list_key_student,
                open(join(data_path_phone_embedding_model, 'list_key_extra_student.pkl'), 'wb'), protocol=2)


def getTeacherStudentAudio():
    """retrieve the audio of each phoneme, and save them into .wav"""
    trainNacta2017_teacher, trainNacta_teacher, trainSepa_teacher, trainPrimarySchool_teacher = getTeacherRecordings()
    valPrimarySchool_student, trainPrimarySchool_student = getStudentRecordings()

    dic_audio_nacta2017_teacher = dumpAudioPhn(wav_path=nacta2017_wav_path,
                                               textgrid_path=nacta2017_textgrid_path,
                                               recordings=trainNacta2017_teacher,
                                               lineTierName='line',
                                               phonemeTierName='details')

    dic_audio_nacta_teacher = dumpAudioPhn(wav_path=nacta_wav_path,
                                           textgrid_path=nacta_textgrid_path,
                                           recordings=trainNacta_teacher,
                                           lineTierName='line',
                                           phonemeTierName='details')

    dic_audio_primarySchool_teacher = dumpAudioPhn(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   recordings=trainPrimarySchool_teacher,
                                                   lineTierName='line',
                                                   phonemeTierName='details')

    dic_audio_sepa_teacher = dumpAudioPhn(wav_path=nacta_wav_path,
                                          textgrid_path=nacta_textgrid_path,
                                          recordings=trainSepa_teacher,
                                          lineTierName='line',
                                          phonemeTierName='details')

    dic_audio_primarySchool_student = dumpAudioPhn(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   recordings=valPrimarySchool_student+trainPrimarySchool_student,
                                                   lineTierName='line',
                                                   phonemeTierName='details')

    dic_audio_teacher = {}
    # fuse two dictionaries
    list_key_teacher = list(set(list(dic_audio_nacta_teacher.keys()) + list(dic_audio_nacta2017_teacher.keys()) +
                            list(dic_audio_primarySchool_teacher.keys()) + list(dic_audio_sepa_teacher.keys())))
    list_key_student = list(set(list(dic_audio_primarySchool_student.keys())))

    for key in list_key_teacher:
        dic_audio_teacher[key] = dic_audio_nacta2017_teacher[key] + dic_audio_nacta_teacher[key] + \
                                 dic_audio_primarySchool_teacher[key] + dic_audio_sepa_teacher[key]
        for ii, phn in enumerate(dic_audio_teacher[key]):
            sf.write(join(phn_wav_path, "teacher", key+"_teacher_"+str(ii)+".wav"), phn, fs)

    for key in list_key_student:
        for ii, phn in enumerate(dic_audio_primarySchool_student[key]):
            sf.write(join(phn_wav_path, "student", key+"_student_"+str(ii)+".wav"), phn, fs)


def getExtraTestAudio():
    """retrieve the audio of each phoneme, and save them into .wav"""
    extra_test_adult = getExtraStudentRecordings()

    dic_audio_extra_adult = dumpAudioPhn(wav_path=primarySchool_wav_path,
                                         textgrid_path=primarySchool_textgrid_path,
                                         recordings=extra_test_adult,
                                         lineTierName='line',
                                         phonemeTierName='details')
    # fuse two dictionaries
    list_key_student = list(set(list(dic_audio_extra_adult.keys())))

    for key in list_key_student:
        for ii, phn in enumerate(dic_audio_extra_adult[key]):
            sf.write(join(phn_wav_path, "extra_test", key+"_extra_test_"+str(ii)+".wav"), phn, fs)


if __name__ == '__main__':
    getTeacherStudentRecordings()
    getExtraTestRecordings()
    getTeacherStudentAudio()
    getExtraTestAudio()