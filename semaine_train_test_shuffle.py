"""
Shuffles Randomly and selects 80% Trainig Data and 20 % Testing data from the vaded semine data and concatenates the vectors
@uthor: Ankit Bahuguna

"""

import os
import cPickle
import numpy as np
import scipy
import random
import shutil
import sys

mfcc_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/mfcc_final_vaded/39D/'
emotion_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/emo_final_vaded/39D/'


#MFCC
mfcc_train_data_dir ='/mnt/alderaan/mlteam3/Assignment3/data/Semine/train/mfcc/39D/'
mfcc_test_data_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/test/mfcc/39D/'

#EMOTION
emo_train_data_dir ='/mnt/alderaan/mlteam3/Assignment3/data/Semine/train/emotion/39D/'
emo_test_data_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/test/emotion/39D/'

#Concatenated
concatenated_dir_train = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/concat/train/39D/'
concatenated_dir_test = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/concat/test/39D/'



def unpackMfccVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc

def saveVectorToDisk(mfcc_vector_output_file, speech_vector_final):
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()

def filtr(files, filetype):
    """Filters a file list by the given filename ending to be accepted"""
    return filter(lambda d: 1 if d.endswith(filetype) else 0, files)

def checkedMakeDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def concatenateVectorsInDirectory(dir_path_input, dir_path_output, filename, num_dimension):
    vector_final = np.zeros((1,num_dimension))
    vector_final = np.delete(vector_final, (0), axis=0)
    OutputFilePath = os.path.join(dir_path_output,filename)
    for root, dirs, files in os.walk(dir_path_input):
        path = root.split('/')
        for file in files:
            FilePath = os.path.join(dir_path_input,str(file))
            mfcc_vector = unpackMfccVector(FilePath)
            vector_final = np.vstack((vector_final,mfcc_vector))
    saveVectorToDisk(OutputFilePath, vector_final)

def concatenateEmotionVectorsInDirectory(dir_path_input, dir_path_output, filename):
    emotion=[]
    OutputFilePath = os.path.join(dir_path_output,filename)
    for root, dirs, files in os.walk(dir_path_input):
        path = root.split('/')
        for file in files:
            FilePath = os.path.join(dir_path_input,str(file))
            mfcc_vector = unpackMfccVector(FilePath)
            mfcc_vector = [item for sublist in mfcc_vector for item in sublist]
            emotion.append(mfcc_vector)

    emotion_final = [int(item) for sublist in emotion for item in sublist]
    saveVectorToDisk(OutputFilePath, emotion_final)

def main():
    checkedMakeDir(mfcc_train_data_dir)
    checkedMakeDir(mfcc_test_data_dir)
    checkedMakeDir(emo_train_data_dir)
    checkedMakeDir(emo_test_data_dir)
    checkedMakeDir(concatenated_dir_train)
    checkedMakeDir(concatenated_dir_test)


    mfcc_file_list = filtr(os.listdir(mfcc_dir),'.dat')
    emotion_file_list = filtr(os.listdir(emotion_dir),'.dat')

    #shuffled_mfcc_list =  random.shuffle(mfcc_file_list)

    random.shuffle(mfcc_file_list)

    #print str(type(mfcc_file_list[1]))
    print str(len(mfcc_file_list))

    #train_set = shuffled_mfcc_list[:48]
    #test_set =  shuffled_mfcc_list[48:]

    train_set = mfcc_file_list[:48]
    test_set =  mfcc_file_list[48:]

    #Copy Training Data (80%)
    for file in train_set:
        mfccfilepath = os.path.join(mfcc_dir,file)
        emotionfilepath = os.path.join(emotion_dir,file)
        shutil.copy(mfccfilepath, mfcc_train_data_dir)
        shutil.copy(emotionfilepath, emo_train_data_dir)

    #Copy Testing Data
    for file in test_set:
        mfccfilepath = os.path.join(mfcc_dir,file)
        emotionfilepath = os.path.join(emotion_dir,file)
        shutil.copy(mfccfilepath, mfcc_test_data_dir)
        shutil.copy(emotionfilepath, emo_test_data_dir)

    #
    # Concatenate Train and Test Set Vectors and Save in the concatenated directory
    #
    concatenateVectorsInDirectory(mfcc_train_data_dir, concatenated_dir_train, 'mfcc_39D.dat', int(39))
    concatenateVectorsInDirectory(mfcc_test_data_dir, concatenated_dir_test, 'mfcc_39D.dat', int(39))
    concatenateEmotionVectorsInDirectory(emo_train_data_dir, concatenated_dir_train, 'emotion_39D.dat')
    concatenateEmotionVectorsInDirectory(emo_test_data_dir, concatenated_dir_test, 'emotion_39D.dat')

    print "Done!"


if __name__== "__main__":
    main()



