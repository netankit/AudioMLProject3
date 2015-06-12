import sys, os, math
import scipy.sparse
import numpy
import cPickle
import scipy
from sklearn import preprocessing

testing_path = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data_copy/test/'

output_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data_copy/test_multiple_fold250/'
def normalize(data):
    return preprocessing.normalize(data,norm='l2')

def read_mfcc_features(input_features):
    with open(input_features, 'rb') as file:
        features = cPickle.load(file)
    features = scipy.sparse.coo_matrix((features),dtype=numpy.float64).toarray()
    if features.shape[1] != 0:
        features = normalize(features)
    return features

def saveVectorToDisk(mfcc_vector_output_file, speech_vector_final):
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()

total_folds = 250

for root, dir, files in os.walk(testing_path):
    for file in files:
        print 'Current File: '+str(file)
        speechFilePath = os.path.join(root,str(file))
        tmp = os.path.dirname(speechFilePath)
        root_dir_name = os.path.basename(tmp)

        probe_mfcc = read_mfcc_features(os.path.join(testing_path, root_dir_name, file)) #angry_test.dat

        fold_index= [None] * int(total_folds)
        size = float(len(probe_mfcc[:]))
        fold_index[0]=0
        for i in xrange(1, (total_folds)):
            fold_index[i] = math.floor((size*i)/float(total_folds))
        emotion_features = probe_mfcc
        for indexnum in xrange(0, len(fold_index)):
            if indexnum is not (len(fold_index)-1):
                emotion_features_fold = emotion_features[fold_index[indexnum]:fold_index[indexnum+1]][:]
            else:
                emotion_features_fold = emotion_features[fold_index[indexnum]:][:]
            outdir = os.path.join(output_dir, root_dir_name)
            outputfile = os.path.join(outdir, str(root_dir_name)+'_fold_'+str(indexnum)+'.dat')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            saveVectorToDisk(outputfile,emotion_features_fold)
print 'Done!'





