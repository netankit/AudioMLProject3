'''
Semine MFCC and Class Label VADing to remove the frames where our svm based classifier has not detected any speech signal
@uthor: Ankit Bahuguna

'''

import numpy as np
import time
import sys
import cPickle
import scipy.sparse
import os

'''
# Uncomment this portion if you want to use this script for multiple parameters

if len(sys.argv)!=6:
    print '\nUsage: python semine_class_label_emotion_vaded.py <semaine_mfcc_dir> <semaine_mfcc_dir> <vad_label_dir> <output_vaded_mfcc_path>  <output_vaded_emotion_path>'
    sys.exit()

semaine_mfcc_dir = sys.argv[1]
semaine_label_dir =  sys.argv[2]
vad_label_dir = sys.argv[3]
output_vaded_mfcc_path = sys.argv[4]
output_vaded_emotion_path = sys.argv[5]

'''
semaine_mfcc_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/mfcc_final/39D/'
semaine_label_dir =  '/mnt/alderaan/mlteam3/Assignment3/data/Semine/emo_final/39D/'
vad_label_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/vad_predicted_mfcc/'

output_vaded_mfcc_path = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/mfcc_final_vaded/39D/'
output_vaded_emotion_path = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/emo_final_vaded/39D'

if not os.path.exists(output_vaded_mfcc_path):
    os.makedirs(output_vaded_mfcc_path)

if not os.path.exists(output_vaded_emotion_path):
    os.makedirs(output_vaded_emotion_path)

def unpackMfccVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc

def unpackMfccStringVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    #print "*** "+str(mfcc)
    mfcc = np.asarray(mfcc)
    return mfcc


def saveVectorToDisk(mfcc_vector_output_file, speech_vector_final):
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()


for root, dirs, files in os.walk(semaine_mfcc_dir):
    path = root.split('/')
    for file in files:
        print "Current File: " + str(file)
        name  = file.replace('.dat','')
        mfcc_file_path = os.path.join(semaine_mfcc_dir, file)
        emotion_file_path = os.path.join(semaine_label_dir, file)
        vad_label_path = os.path.join(vad_label_dir, str(name)+'_predicted.dat')

        mfcc = unpackMfccVector(mfcc_file_path)
        emotion_vec = unpackMfccVector(emotion_file_path)
        vad_label_vec = unpackMfccStringVector(vad_label_path)

        print "Original MFCC Shape: "+str(mfcc.shape)

        emotion = emotion_vec.tolist()
        emotion = [item for sublist in emotion for item in sublist]
        label = vad_label_vec.tolist()


        label = map(str, label)
        emotion = map(str, emotion)
        label = [float(i.strip('[').strip(']')) for i in label]
        emotion = [(i.strip('[').strip(']')) for i in emotion]

        ones_index = [i for i, j in enumerate(label) if j == 1]
        ones_mfcc = [mfcc[i] for i in ones_index]
        ones_emotion  = [emotion[i] for i in ones_index]

        print "Num. Speech Frames : "+str(len(ones_index))
        print "Final MFCC Shape: "+str(len(ones_mfcc))
        print "Emotion: "+str(len(ones_emotion))

        mfcc_file_output_path = os.path.join(output_vaded_mfcc_path, file)
        emotion_file_output_path = os.path.join(output_vaded_emotion_path, file)

        saveVectorToDisk(mfcc_file_output_path, ones_mfcc)
        saveVectorToDisk(emotion_file_output_path ,ones_emotion)
print "Done"

