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

if len(sys.argv)!=2:
    print '\nUsage: python scoring_GMM_UBM_final.py <number_of_gaussians> '
    sys.exit()

gaussians = sys.argv[1]
variance_threshold = 5e-4
# INPUT DIRECTORY
training_path = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_emotion_model_TIMIT_FAU_UBM_balanced/39D/' + gaussians
testing_path = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/39D/mfcc_segmented/Mont_testing'

emotions = {'N','E','A','M'}

# read UBM
'''
# TIMIT 13D
if (gaussians == '64'):
	ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/gmm_ubm_64G.hdf5'
if (gaussians == '128'):
	ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/gmm_ubm_128G.hdf5'
if (gaussians == '256'):
	ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/gmm_ubm_256G.hdf5'
if (gaussians == '512'):

# TIMIT 26D
if (gaussians == '64'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_64G.hdf5'
if (gaussians == '128'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_128G.hdf5'
if (gaussians == '256'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_256G.hdf5'
if (gaussians == '512'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_512G.hdf5'
'''
# TIMIT with FAU no label UBM
if (gaussians == '64'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_ubm_64G.hdf5'
if (gaussians == '128'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_ubm_128G.hdf5'
if (gaussians == '256'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_ubm_256G.hdf5'
if (gaussians == '512'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_512G.hdf5'

ubm_hdf5 = bob.io.HDF5File(ubm_file)
ubm = bob.machine.GMMMachine(ubm_hdf5)
ubm.set_variance_thresholds(variance_threshold)

def load_features(model_features_input):
	# read MFCC Features
	with open(model_features_input, 'rb') as infile:
		model_features = cPickle.load(infile)
	infile.close()
	model_features_arr = scipy.sparse.coo_matrix((model_features), dtype=numpy.float64).toarray()
	return model_features_arr

def normalize(data):
    return preprocessing.normalize(data,norm='l2')

accuracy_list = []
# cosine_score
for emotion_probe in emotions:
	correct = 0.0
	wrong = 0.0
	emotion_path = os.path.join(testing_path, emotion_probe)
	emotion_features = numpy.empty([0,39])
	for root, dir, files in os.walk(emotion_path):
		for file in files:
			probe_mfcc = load_features(os.path.join(root, file))
			emotion_features = numpy.vstack([emotion_features, probe_mfcc])
			normalize(emotion_features)
			probe_score = 0.0
			probe_predict = ''
			for emotion_model in emotions:
				model = bob.machine.GMMMachine(bob.io.HDF5File(os.path.join(training_path, emotion_model+'.hdf5')))
				score = numpy.mean([model.log_likelihood(emotion_feature) - ubm.log_likelihood(emotion_feature)
				for emotion_feature in emotion_features])
				if probe_score < score:
					probe_score = score
					probe_predict = emotion_model
				print emotion_probe, emotion_model, score
			if probe_predict == emotion_probe:
				correct = correct + 1
			else:
				wrong = wrong + 1
			print correct, wrong
	accuracy = correct / ( correct + wrong)
	accuracy_list += [(emotion_probe, accuracy)]

df = pandas.DataFrame(accuracy_list)
df.columns = ["probe", "accuracy"]
df.to_csv("results/gmmubm_" + gaussians+" G_39D_balanced_extended.csv")
