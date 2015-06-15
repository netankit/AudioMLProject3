'''
Concatenates vertically the mfcc for the ubm dataset with the FAU dataset wav's which are not labelled with emotion.
Used for further generating larger UBMS.
Dimension [MFCC] : 13, 26 and 39
@uthor: Ankit Bahuguna

'''

import scipy.io.wavfile as wav
import numpy as np
import math
from features import mfcc
import sys
import os
import cPickle
import scipy.sparse

def unpackMfccVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc

print "Start.."
ubm_13D_path = '/mnt/alderaan/mlteam3/Assignment2/data/mfcc_speechonly_vaded/ubm_mfcc.dat'
wav_nolabel_13D_path = '/mnt/alderaan/mlteam3/Assignment3/data/no_label_chunks_mfcc_nonsegmented/mfcc_seg_nolabel_13D.dat'

ubm_26D_path = '/mnt/alderaan/mlteam3/Assignment2/data/mfcc_speechonly_vaded/ubm_mfcc_26D.dat'
wav_nolabel_26D_path = '/mnt/alderaan/mlteam3/Assignment3/data/no_label_chunks_mfcc_nonsegmented/mfcc_seg_nolabel_26D.dat'

ubm_39D_path = '/mnt/alderaan/mlteam3/Assignment2/data/mfcc_speechonly_vaded/ubm_mfcc_39D.dat'
wav_nolabel_39D_path = '/mnt/alderaan/mlteam3/Assignment3/data/no_label_chunks_mfcc_nonsegmented/mfcc_seg_nolabel_39D.dat'


#ubm_13D = unpackMfccVector(ubm_13D_path)
#ubm_26D = unpackMfccVector(ubm_26D_path)

ubm_39D = unpackMfccVector(ubm_39D_path)

#wav_13D = unpackMfccVector(wav_nolabel_13D_path)
#wav_26D = unpackMfccVector(wav_nolabel_26D_path)

wav_39D = unpackMfccVector(wav_nolabel_39D_path)


#final_13D = np.vstack((ubm_13D,wav_13D))
#final_26D = np.vstack((ubm_26D,wav_26D))
final_39D = np.vstack((ubm_39D,wav_39D))

'''
mfcc_vector_output_file1 = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm/mfcc_vaded/final_mfcc_13D.dat'
mfcc_vector_file1 = open(mfcc_vector_output_file1, 'w')
temp1 = scipy.sparse.coo_matrix(final_13D)
print temp1.shape
cPickle.dump(temp1,mfcc_vector_file1,-1)
mfcc_vector_file1.close()
print "13d file written"

mfcc_vector_output_file2 = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm/mfcc_vaded/final_mfcc_26D.dat'
mfcc_vector_file2 = open(mfcc_vector_output_file2, 'w')
temp1 = scipy.sparse.coo_matrix(final_26D)
print temp1.shape
cPickle.dump(temp1,mfcc_vector_file2,-1)
mfcc_vector_file2.close()
print "26d file written"

'''

mfcc_vector_output_file2 = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_mfcc/mfcc_vaded/final_mfcc_39D.dat'
mfcc_vector_file2 = open(mfcc_vector_output_file2, 'w')
temp1 = scipy.sparse.coo_matrix(final_39D)
print temp1.shape
cPickle.dump(temp1,mfcc_vector_file2,-1)
mfcc_vector_file2.close()
print "39d file written"
