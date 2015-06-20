import sys, os, shutil
import argparse
import bob
import numpy
import cPickle
import scipy.sparse
import facereclib.utils as utils
from sklearn import preprocessing
'''
KmeansUBM.py
------------

It performs the KMeans + GMM Training (Assignment 3)
'''

if len(sys.argv)!=5:
    print '\nUsage: python KmeansUBM.py <number_of_gaussians> <Input_speechonly_mfcc_filepath> <kmeans_hdf5_filename> <gmm_hdf5_filename>'
    sys.exit()

# Command Line Arguments
number_of_gaussians = int(sys.argv[1])
Input_speechonly_mfcc_filepath = sys.argv[2]
kmeans_hdf5_filename = sys.argv[3]
gmm_hdf5_filename = sys.argv[4]

# parameters for the GMM
gaussians = number_of_gaussians
max_iterations = 500
training_threshold = 5e-4
variance_threshold = 5e-4
INIT_SEED = 2015

KMeans_HDF5 = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/'+str(kmeans_hdf5_filename).strip()
GMM_HDF5 = '/mnt/alderaan/mlteam3/Assignment3/data/giantubm_model/gmm_ubm/39D/'+str(gmm_hdf5_filename).strip()
InputData = Input_speechonly_mfcc_filepath

def normalize(data):
	return preprocessing.normalize(data,norm='l2')


def kmeans(data):
	"""the K-Means training."""
	# read data
	print "UBM Training - Step 1: initializing kmeans"
	output_file = KMeans_HDF5
	# Perform KMeans initialization
	kmeans_machine = bob.machine.KMeansMachine(gaussians, data.shape[1])
	# Creates the KMeansTrainer and trains the Kmeans
	kmeans_trainer = bob.trainer.KMeansTrainer()
	kmeans_trainer.initialization_method = kmeans_trainer.initialization_method_type.RANDOM_NO_DUPLICATE
	kmeans_trainer.max_iterations =  max_iterations
	kmeans_trainer.convergence_threshold = variance_threshold
	kmeans_trainer.rng = bob.core.random.mt19937(INIT_SEED)

	kmeans_trainer.train(kmeans_machine, data)
	utils.ensure_dir(os.path.dirname(output_file))
	kmeans_machine.save(bob.io.HDF5File(output_file, 'w'))
	print "UBM Training - Step 1: Saved KMeans machine to '%s'" % output_file

def gmm(data):
	"""Initializes the GMM calculation with the result of the K-Means algorithm (non-parallel).
	 This might require a lot of memory."""
	output_file = GMM_HDF5
	print "UBM Training - Step 2: Initializing GMM...."

	# load KMeans machine
	kmeans_machine = bob.machine.KMeansMachine(bob.io.HDF5File(KMeans_HDF5))

	# Create initial GMM Machine
	gmm_machine = bob.machine.GMMMachine(gaussians, data.shape[1])

	[variances, weights] = kmeans_machine.get_variances_and_weights_for_each_cluster(data)

	# Initializes the GMM
	gmm_machine.means = kmeans_machine.means
	gmm_machine.variances = variances
	gmm_machine.weights = weights
	gmm_machine.set_variance_thresholds(variance_threshold)

	# Creates the GMMTrainer and trains the GMM
	gmm_trainer = bob.trainer.ML_GMMTrainer(True, True, True)
	gmm_trainer.max_iterations = max_iterations
	gmm_trainer.rng = bob.core.random.mt19937(INIT_SEED)

	gmm_trainer.train(gmm_machine, data)
	utils.ensure_dir(os.path.dirname(output_file))
	gmm_machine.save(bob.io.HDF5File(os.path.join(output_file), 'w'))
	print "UBM Training - Step 2: Wrote GMM file '%s'" % output_file


with open(InputData, 'rb') as infile1:
	data = cPickle.load(infile1)
data = scipy.sparse.coo_matrix((data),dtype=numpy.float64).toarray()



# Script -- Main Function
data = normalize(data)
kmeans(data)
gmm(data)
print 'UBM Training via Kmeans based GMM Training Finished.'