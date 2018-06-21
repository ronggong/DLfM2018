import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.filePath import *
from src.parameters import *
from src.phonemeMap import dic_pho_map
from src.textgridParser import syllableTextgridExtraction
from src.train_test_filenames import getTeacherRecordings
from src.train_test_filenames import getStudentRecordings
from src.train_test_filenames import getExtraStudentRecordings
import soundfile as sf


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
    getTeacherStudentAudio()
    getExtraTestAudio()