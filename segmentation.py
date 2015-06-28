import numpy as np
import time
import sys
import cPickle
import scipy.sparse
import os
from sklearn import preprocessing

original_speech_path = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/39D/noisy_mfcc/wav'
segmentation_path = '/mnt/tatooine/data/emotion/FAU-Aibo-Emotion-Corpus/segmentation/segmentation.txt'
labels_path = '/mnt/tatooine/data/emotion/FAU-Aibo-Emotion-Corpus/labels/CEICES/chunk_labels_4cl_aibo_chunk_set.txt'

output_segmented_path = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/39D/mfcc_segmented'

segmentations = np.loadtxt(segmentation_path,dtype='str')
labels = np.loadtxt(labels_path,dtype='str')

total = len(labels)
count = 0
for label in labels:
    label_name, emotion, confidence = label
    contents = []
    starts_ends = []
    for segmentation in segmentations:
        seg_name, content, start, end = segmentation
        if seg_name.find(label_name) != -1:
            contents.append(content)
            start_end = start, end
            starts_ends.append(start_end)
    school = label_name.split('_')[0]
    for root, dir, files in os.walk(original_speech_path,topdown=True):
        for file in files:
            speech_name = file[5:-8]
            if speech_name.find(label_name) != -1:
                mfcc_file = os.path.join(root, str(file))
                with open(mfcc_file, 'rb') as infile1:
                    mfcc = cPickle.load(infile1)
                infile1.close()
                mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
                mfcc_segmented = np.empty([0,39])
                for start_end in starts_ends:
                    start, end = start_end
                    mfcc_segmented = np.vstack((mfcc_segmented, mfcc[int(start)-1:int(end)-1]))
                filedir = output_segmented_path + '/' + school + '/' + emotion + '/' 
                if not os.path.exists(filedir):
                    os.makedirs(filedir)
                filepath = filedir + file[5:-8] + '_segmented.dat'
                mfcc_segmented_file = open(filepath,'w')
                temp = scipy.sparse.coo_matrix(mfcc_segmented)
                cPickle.dump(temp, mfcc_segmented_file, -1)
                mfcc_segmented_file.close()
                count += 1
                print 'Total:', total, 'Processing', count
