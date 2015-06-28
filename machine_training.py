import os
import bob
import numpy

ivector_file = '/mnt/alderaan/mlteam3/Assignment3/ivectors_100/39D/128/ubm_noisy_mfcc_39D'
whitening_enroler_file = 'model/whitening.hdf5'
wccn_enroller_file = 'model/wccn.hdf5'
lda_enroller_file = 'model/lda.hdf5'
pca_enroller_file = 'model/pca_50_20_39D.hdf5'
plda_enroller_file = 'model/plda_50_20_39D.hdf5'

SUBSPACE_DIMENSION_OF_F = 50
SUBSPACE_DIMENSION_OF_G = 20


def read_ivectors(ivector_file):
	ivectors_matrix = []
	for root, dir, files in os.walk(ivector_file):
	    for file in files:
	        ivec_path = os.path.join(root, str(file))
	        ivec = bob.io.HDF5File(ivec_path)
	        ivector = ivec.read('ivec')
	        ivector = numpy.array(ivector)
	        #ivectors_matrix = numpy.append(ivectors_matrix, ivector)
	        ivectors_matrix.append(ivector)
	#ivectors_matrix = numpy.vstack(ivectors_matrix)

	#ivectors_matrix = ivectors_matrix.reshape(len(ivectors_matrix)/400, 400)
	return ivectors_matrix

def read_ivectors_whiten(ivector_file):
	ivectors_matrix = []
	for root, dir, files in os.walk(ivector_file):
	    for file in files:
	        ivec_path = os.path.join(root, str(file))
	        ivec = bob.io.HDF5File(ivec_path)
	        ivector = ivec.read('ivec')
	        ivectors_matrix.append(ivector)
	ivectors_matrix = numpy.vstack(ivectors_matrix)

	#ivectors_matrix = ivectors_matrix.reshape(len(ivectors_matrix)/400, 400)
	return ivectors_matrix

def train_whitening_enroller(train_files):
	""" i-vector preprocessing: training whitening enroller"""

	ivectors_matrix = read_ivectors_whiten(train_files)
	# create a Linear Machine     # Runs whitening (first method)
	whitening_machine = bob.machine.LinearMachine(ivectors_matrix.shape[1],ivectors_matrix.shape[1])

	# create the whitening trainer
	t = bob.trainer.WhiteningTrainer()

	t.train(whitening_machine, ivectors_matrix)

	# Save the whitening linear machine
	print("Saving the whitening machine..")
	whitening_machine.save(bob.io.HDF5File(whitening_enroler_file, "w"))

	return whitening_machine

def project_whitening(whitening_machine, ivector_file):
	""" i-vector preprocessing: projecting whitening """
	whitened_ivectors = []
	for root, dir, files in os.walk(ivector_file):
	    for file in files:
	    	speaker_ivectors = []
	        ivec_path = os.path.join(root, str(file))
	        ivec = bob.io.HDF5File(ivec_path)
	        ivectors = ivec.read('ivec')
	        for ivector in ivectors:
	            whitened_ivector = whitening_machine.forward(ivector)
	            normalized_ivector = whitened_ivector/numpy.linalg.norm(whitened_ivector)
	            speaker_ivectors.append(normalized_ivector)
	        whitened_ivectors.append(speaker_ivectors)     
	#ivectors_matrix = numpy.vstack(ivectors_matrix)

	#ivectors_matrix = ivectors_matrix.reshape(len(ivectors_matrix)/400, 400)
	return whitened_ivectors

def project_whitening_ivec(whitening_machine, ivec_path):
	""" i-vector preprocessing: projecting whitening """
	speaker_ivectors = []
	ivec = bob.io.HDF5File(ivec_path)
	ivectors = ivec.read('ivec')
	for ivector in ivectors:
	    whitened_ivector = whitening_machine.forward(ivector)
	    normalized_ivector = whitened_ivector/numpy.linalg.norm(whitened_ivector)
	    speaker_ivectors.append(normalized_ivector) 

	return speaker_ivectors

def train_lda_enroller(ivectors_matrix):
	""" i-vector preprocessing: training lda enroller"""

	# create the FisherLDATrainer
	t = bob.trainer.FisherLDATrainer(strip_to_rank=False)

	LDA_machine, __eig_vals = t.train(ivectors_matrix)

	# Save the whitening linear machine
	print("Saving the LDA machine..")
	LDA_machine.save(bob.io.HDF5File(lda_enroller_file, "w"))

	return LDA_machine

def project_lda(LDA_machine, sample):
	""" i-vector preprocessing: projecting lda """
	projected_sample =  LDA_machine.forward(sample)
	return projected_sample

def train_wccn_enroller(train_files):
	""" i-vector preprocessing: training Within-Class Covariance Normalisation enroller"""
	ivectors_matrix = read_ivectors(train_files)
	#print type(ivectors_matrix[1])
	#print ivectors_matrix[1].shape
	# create the whitening trainer
	t = bob.trainer.WCCNTrainer()
	# Trains the LinearMachine to perform the WCCN, given a training set.
	wccn_machine = t.train(ivectors_matrix)
	
	# Save the whitening linear machine
	print("Saving the wccn machine..")
	wccn_machine.save(bob.io.HDF5File(wccn_enroller_file, "w"))

	return wccn_machine


def train_pca(training_features):
	"""Trains and returns a LinearMachine that is trained using PCA"""
	data_list = []
	for client in training_features:
		for feature in client:
			data_list.append(feature)
	data = numpy.vstack(data_list)
	t = bob.trainer.PCATrainer()
	machine, __eig_vals = t.train(data)
	# limit number of pcs
	# machine.resize(machine.shape[0], subspace_dimension_pca)
	return machine

def perform_pca_client(machine, client):
	"""Perform PCA on an array"""
	client_data_list = []
	for feature in client:
		# project data
		projected_feature = numpy.ndarray(machine.shape[1], numpy.float64)
		machine(feature, projected_feature)
		# add data in new array
		client_data_list.append(projected_feature)
	client_data = numpy.vstack(client_data_list)

	return client_data

def perform_pca(machine, training_set):
	"""Perform PCA on data"""
	data = []
	for client in training_set:
		client_data = perform_pca_client(machine, client)
		data.append(client_data)
	return data

def train_plda_enroller(train_files):

	# load GMM stats from training files
	training_features = read_ivectors(train_files)

	# train PCA and perform PCA on training data
	pca_machine = train_pca(training_features)
	training_features = perform_pca(pca_machine, training_features)

	input_dimension = training_features[0].shape[1]

	print("Training PLDA base machine")
	# create trainer
	t = bob.trainer.PLDATrainer()
	# train machine
	plda_base = bob.machine.PLDABase(input_dimension, SUBSPACE_DIMENSION_OF_F, SUBSPACE_DIMENSION_OF_G)
	t.train(plda_base, training_features)

	# write machines to file
	proj_hdf5file = bob.io.HDF5File(plda_enroller_file, "w")
	proj_hdf5file.create_group('/pca')
	proj_hdf5file.cd('/pca')
	pca_machine.save(proj_hdf5file)
	proj_hdf5file.create_group('/plda')
	proj_hdf5file.cd('/plda')
	plda_base.save(proj_hdf5file)
	print "saved plda machines"


whitening = train_whitening_enroller(ivector_file)
whitened_ivectors = project_whitening(whitening, ivector_file) # normalized

lda = train_lda_enroller(whitened_ivectors)
data = map(lda.forward, ivectors_matrix)
train_wccn_enroller(ivector_file)
train_plda_enroller(ivector_file)
