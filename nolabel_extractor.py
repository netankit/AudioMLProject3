'''
Extracts the non emotion labeled files from the set of 18K wav files
and concatenates their vectors to be used for training a 13 Dimenstional UBM

@uthor: Ankit Bahuguna
Date: 19.06.2015
'''

import scipy.io.wavfile as wav
import numpy as np
import math
from features import mfcc
import sys
import os
import cPickle
import scipy.sparse

file_list = []

#mfcc_noisy_dir = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/13D/mfcc_segmented/Mont_testing/'
#mfcc_noisy_dir = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/26D/mfcc_whole_segmented/'
mfcc_noisy_dir = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/39D/mfcc_whole_segmented/'
#mfcc_vector_output_file = '/mnt/alderaan/mlteam3/Assignment3/data/no_label_chunks_mfcc_nonsegmented/mfcc_non_seg_nolabel_mont.dat'
#mfcc_vector_output_file = '/mnt/alderaan/mlteam3/Assignment3/data/no_label_chunks_mfcc_nonsegmented/mfcc_seg_nolabel_26D.dat'
mfcc_vector_output_file = '/mnt/alderaan/mlteam3/Assignment3/data/no_label_chunks_mfcc_nonsegmented/mfcc_seg_nolabel_39D.dat'
mfcc_dimensions = 39


def getMfccVector(noise_mix_speech_file):
    (rate, signal) = wav.read(noise_mix_speech_file)
    mfcc_vec = mfcc(signal,rate,winlen=0.025,winstep=0.01,numcep=mfcc_dimensions,
          nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          ceplifter=22,appendEnergy=True)
    return mfcc_vec

def unpackMfccVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc


def createFileList():
    with open('/mnt/tatooine/data/emotion/FAU-Aibo-Emotion-Corpus/labels/CEICES/chunk_labels_4cl_aibo_chunk_set.txt') as f1:
        for line1 in f1:
            filename = line1.split(' ')
            tmp = str(filename[0].rstrip('\n'))+'_segmented.dat'
            file_list.append(tmp)
    f1.close()


def main():
    createFileList()

    speech_vector_final = np.zeros((1,mfcc_dimensions))
    speech_vector_final = np.delete(speech_vector_final, (0), axis=0)
    count =int(0)
    for root, dirs, files in os.walk(mfcc_noisy_dir):
            path = root.split('/')
            for file in files:
                print "File Read: "+str(file)
                if not str(file) in file_list:
                    print "Count: "+str(count)
                    speechFilePath = os.path.join(root,str(file))
                    tmp = os.path.dirname(speechFilePath)
                    root_dir_name = os.path.basename(tmp)
                    filepath = os.path.join(mfcc_noisy_dir,root_dir_name,file)
                    print "File: "+str(filepath)
                    mfcc_vector = unpackMfccVector(filepath)
                    speech_vector_final = np.vstack((speech_vector_final,mfcc_vector))
                    count+=1
                else:
                    print "False"

    #if not os.path.exists(mfcc_vector_output_file):
    #   os.makedirs(mfcc_vector_output_file)
    #Write the mfcc speech vector for speech file.
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    print temp1.shape
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()

    print "Finished!"
    #print file_list

if __name__ == "__main__":
    main()

