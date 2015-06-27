import os
import sys
import bob
import numpy
import shutil
import cPickle
import scipy.sparse
import facereclib.utils as utils
from sklearn import preprocessing

if len(sys.argv)!=2:
    print '\nUsage: python project_ivectors_semaine.py <number_of_gaussians> '
    sys.exit()

# parameters
training_threshold = 5e-4
variance_threshold = 5e-4
max_iterations = 25
relevance_factor = 4             # Relevance factor as described in Reynolds paper
gmm_enroll_iterations = 1        # Number of iterations for the enrollment phase
subspace_dimension_of_t = 100
INIT_SEED = 5489

# parameters for the GMM
gaussians = sys.argv[1]

# TIMIT with FAU no label UBM 39D
if (gaussians == '64'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_64G.hdf5'
if (gaussians == '128'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_128G.hdf5'
if (gaussians == '256'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_256G.hdf5'
if (gaussians == '512'):
    ubm_file = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/gmm_giantubm_512G.hdf5'

input_ubm_features = '/mnt/alderaan/mlteam3/Assignment2/data/Noisy_TIMIT_buckeye/ubm_noisy_mfcc_39D'

'''
input_emotion_train_features = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/train_test_data_vaded/train/mfcc/39D/'
input_emotion_test_features = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/train_test_data_vaded/test/mfcc/39D'
'''

input_emotion_train_features = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data_copy/train/'
input_emotion_test_features = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data_copy/test/'

def normalize(data):
    return preprocessing.normalize(data,norm='l2')

def read_mfcc_features(input_features):
    with open(input_features, 'rb') as file:
        features = cPickle.load(file)
    features = scipy.sparse.coo_matrix((features),dtype=numpy.float64).toarray()
    if features.shape[1] != 0:
        features = normalize(features)
    return features

def load_training_gmmstats(input_features):

    gmm_stats_list = []
    for root, dir, files in os.walk(input_features):
        for file in files:
            features_path = os.path.join(root, str(file))
            features = read_mfcc_features(features_path)
            stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
            if features.shape[1] == 39:
                ubm.acc_statistics(features, stats)
                gmm_stats_list.append(stats)

    return gmm_stats_list

def train_enroller(input_features):

    # load GMM stats from UBM training files
    gmm_stats = load_training_gmmstats(input_features)

    # Training IVector enroller
    output_file = 'data/tv_enroller_512G_39D_25i_100.hdf5'

    print "training enroller (total variability matrix) ", max_iterations, 'max_iterations'
    # Perform IVector initialization with the UBM
    ivector_machine = bob.machine.IVectorMachine(ubm, subspace_dimension_of_t)
    ivector_machine.variance_threshold = variance_threshold

    # Creates the IVectorTrainer and trains the ivector machine
    ivector_trainer = bob.trainer.IVectorTrainer(update_sigma=True, convergence_threshold=variance_threshold, max_iterations=max_iterations)
    # An trainer to extract i-vector (i.e. for training the Total Variability matrix)
    ivector_trainer.train(ivector_machine, gmm_stats)
    ivector_machine.save(bob.io.HDF5File(output_file, 'w'))
    print "IVector training: saved enroller's IVector machine base to '%s'" % output_file

    return ivector_machine

def lnorm_ivector(ivector):
    norm = numpy.linalg.norm(ivector)
    if norm != 0:
        return ivector/numpy.linalg.norm(ivector)
    else:
        return ivector

def save_ivectors(data, feature_file):
    hdf5file = bob.io.HDF5File(feature_file, "w")
    hdf5file.set('ivec', data)

def project_ivectors(input_features):
    """Extract the ivectors for all files of the database"""
    print "projecting ivetors"
    tv_enroller = bob.machine.IVectorMachine(ubm, subspace_dimension_of_t)
    tv_enroller.load(bob.io.HDF5File("/mnt/alderaan/mlteam3/Assignment3/data/tv_enroller_512G_39D_25i_100.hdf5"))
    #print input_features
    for root, dir, files in os.walk(input_features):
        ivectors = []
        for file in files:
            features_path = os.path.join(root, str(file))
            features = read_mfcc_features(features_path)
            stats = bob.machine.GMMStats(ubm.dim_c, ubm.dim_d)
            if features.shape[1] == 39:
                ubm.acc_statistics(features, stats)
                ivector = tv_enroller.forward(stats)
                lnorm_ivector(ivector)
                ivectors.append(ivector)
        ivectors_path = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/ivectors_100_emotionwise/test/'+ gaussians + '/' + input_features.split('/')[-1]
        if not os.path.exists(ivectors_path):
        	os.makedirs(ivectors_path)
        ivectors_path =	ivectors_path + '/' + os.path.split(root)[1] + '_' + gaussians + '_100.ivec'
        save_ivectors(ivectors, ivectors_path)
        print "saved ivetors to '%s' " % ivectors_path


#############################################
ubm_hdf5 = bob.io.HDF5File(ubm_file)
ubm = bob.machine.GMMMachine(ubm_hdf5)
ubm.set_variance_thresholds(variance_threshold)

#train_enroller(input_ubm_features)
#project_ivectors(input_ubm_features)
#project_ivectors(input_emotion_train_features)
project_ivectors(input_emotion_test_features)


