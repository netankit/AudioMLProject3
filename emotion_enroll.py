import bob
import numpy
import cPickle
import scipy.sparse
import random
import os
import sys
from sklearn import preprocessing

if len(sys.argv)!=2:
    print '\nUsage: python scoring_GMM_UBM_final.py <number_of_gaussians> '
    sys.exit()

print "Starting MAP GMM-UBM Enrollment Training for emotions..."
# parameters for the GMM
training_threshold = 5e-4
variance_threshold = 5e-4
# parameters of the GMM enrollment
relevance_factor = 4         # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1    # Number of iterations for the enrollment phase
INIT_SEED = 5489
gaussians = sys.argv[1]

training_dir = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/39D/mfcc_segmented/Ohm_training'
testing_dir = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_mfcc/39D/mfcc_segmented/Mont_testing'

## OUTPUT DIRECTORY
model_output_path = '/mnt/alderaan/mlteam3/Assignment3/data/FAU_emotion_model_TIMIT_FAU_UBM_balanced/39D/' + gaussians
if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)

# read UBM
'''
# TIMIT UBM 26D
if (gaussians == '64'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_64G.hdf5'
if (gaussians == '128'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_128G.hdf5'
if (gaussians == '256'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_256G.hdf5'
if (gaussians == '512'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment2/model_gmm_ubm/26D_ubm/gmm_ubm_26D_512G.hdf5'

'''
# TIMIT with FAU no label UBM 39D
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
    # read MFCC Features
    with open(model_features_input, 'rb') as infile:
        model_features = cPickle.load(infile)
    infile.close()
    model_features_arr = scipy.sparse.coo_matrix((model_features), dtype=numpy.float64).toarray()
    return model_features_arr

def normalize(data):
    return preprocessing.normalize(data,norm='l2')

# prepare MAP_GMM_Trainer
MAP_GMM_trainer = bob.trainer.MAP_GMMTrainer(relevance_factor=relevance_factor, update_means=True, update_variances=False, update_weights=False)
rng = bob.core.random.mt19937(INIT_SEED)
MAP_GMM_trainer.set_prior_gmm(ubm)

emotions = {'N','E','A','M'}

# Enrolls a GMM using MAP adaptation of the UBM, given a list of 2D numpy.ndarray's of feature vectors"""
for emotion in emotions:
    emotion_path = os.path.join(training_dir, emotion)
    emotion_features = numpy.empty([0,39])
    for root, dir, files in os.walk(emotion_path):
    	for file in files[:461]: # balancing four emotion training sets
    		mfcc = load_features(os.path.join(root, file))
    		emotion_features = numpy.vstack([emotion_features, mfcc])
    		normalize(emotion_features)
    output_model_file = os.path.join(model_output_path, emotion+'.hdf5')
    gmm = bob.machine.GMMMachine(ubm)
    gmm.set_variance_thresholds(variance_threshold)
    MAP_GMM_trainer.train(gmm, emotion_features)  #, gmm_enroll_iterations, training_threshold, rng
    gmm.save(bob.io.HDF5File(output_model_file, 'w'))
    print 'saved', emotion
