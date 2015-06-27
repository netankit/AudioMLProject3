'''
GMM_UBM For 39D Semaine Data (Emotionwise)
@uthor: Ankit Bahuguna
+Fold wise Accuracy Calculation for each test probe.
'''

import os
import sys
import bob
import numpy
import math
import cPickle
import scipy.sparse
from itertools import *
import pandas
from sklearn import preprocessing


if len(sys.argv)!=3:
    print '\nUsage: python gmmubm_score_semaine.py <number_of_gaussians> <num_folds> '
    sys.exit()

gaussians = sys.argv[1]
total_folds = int(sys.argv[2])

variance_threshold = 5e-4

# INPUT DIRECTORY
training_path = '/mnt/alderaan/mlteam3/Assignment3/data/semaine_emotion_gmm/39D/gmm_files/' + gaussians
testing_path = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/test/'


emotion_list = {'angry','happy','neutral','relaxed','sad'}


# TIMIT with FAU no label / GIANT-UBM
if (gaussians == '64'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_64G.hdf5'
if (gaussians == '128'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_128G.hdf5'
if (gaussians == '256'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_256G.hdf5'
if (gaussians == '512'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_512G.hdf5'

ubm_hdf5 = bob.io.HDF5File(ubm_file)
ubm = bob.machine.GMMMachine(ubm_hdf5)
ubm.set_variance_thresholds(variance_threshold)

def load_features(model_features_input):
	''' Read and return MFCC Features numpy array'''
	with open(model_features_input, 'rb') as infile:
		model_features = cPickle.load(infile)
	infile.close()
	model_features_arr = scipy.sparse.coo_matrix((model_features), dtype=numpy.float64).toarray()
	return model_features_arr

def normalize(data):
    return preprocessing.normalize(data,norm='l2')



accuracy_list = []
# Log-Likelihood Scoring

for root, dir, files in os.walk(testing_path):
    for file in files:
        probe_mfcc = load_features(os.path.join(testing_path, file)) #angry_test.dat
        emotion_original = file.replace("_test.dat","")

        print 'TEST PROBE: ' +str(emotion_original)
        print "Test Probe Shape: "+str(probe_mfcc.shape)

        fold_index= [None] * int(total_folds)
        size = float(len(probe_mfcc[:]))
        fold_index[0]=0
        for i in xrange(1, (total_folds)):
            fold_index[i] = math.floor((size*i)/float(total_folds))
        #print fold_index



        # Normalize emotion features of test probe by scikit - l2 normalization
        emotion_features = normalize(probe_mfcc)
        correct = 0.0
        wrong = 0.0
        # split the test features into n folds, thus we get fold wise accuray scores
        for indexnum in xrange(0, len(fold_index)):
            print 'Starting Fold: '+str(indexnum)+' ...'
            probe_score_final = 0.0
            probe_predicted_emotion = ''

            for emotion_gmmmodel_value in emotion_list:
                model = bob.machine.GMMMachine(bob.io.HDF5File(os.path.join(training_path, 'gmm_'+str(emotion_gmmmodel_value)+'_gmm_semaine_39D_'+str(gaussians)+'G.hdf5')))
                # Note : Make sure you slice out folds carefully!
                if indexnum is not (len(fold_index)-1):
                    emotion_features_fold = emotion_features[fold_index[indexnum]:fold_index[indexnum+1]][:]
                    #print 'one'
                else:
                    emotion_features_fold = emotion_features[fold_index[indexnum]:][:]
                    #print 'two'
                #print emotion_features_fold

                #print 'Emotion Features Fold Shape' +str(emotion_features_fold.shape)
                #sys.exit()
                score = numpy.mean([model.log_likelihood(emotion_feature) - ubm.log_likelihood(emotion_feature) for emotion_feature in emotion_features_fold])

                if probe_score_final < score:
                    probe_score_final = score
                    probe_predicted_emotion = emotion_gmmmodel_value
            if probe_predicted_emotion == emotion_original:
                correct +=1
            else:
                wrong += 1
            #print correct,wrong

        print "# Correct: "+str(correct)
        print '# Incorrect: '+str(wrong)
        accuracy = correct / (correct + wrong)
        print 'Accuracy for Test File:'+str(emotion_original)+' is: '+str(accuracy)
        accuracy_list += [(emotion_original, accuracy)]

df = pandas.DataFrame(accuracy_list)
df.columns = ["probe", "accuracy"]
df.to_csv("results/semaine_results/semaine_gmmubm_" + gaussians+"G_"+str(total_folds)+"fold_39D.csv")
