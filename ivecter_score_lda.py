import os
import bob
import numpy
import math
from itertools import *
import pandas

# INPUT DIRECTORY
ubm_ivectors_path  = '/mnt/alderaan/mlteam3/Assignment3/ivectors_100/39D/128/ubm_noisy_mfcc_39D'
train_ivectors_path = '/mnt/alderaan/mlteam3/Assignment3/ivectors_100/39D/128/Ohm_training'
test_ivectors_path = '/mnt/alderaan/mlteam3/Assignment3/ivectors_100/39D/128/Mont_testing'

emotion_list = ["A_128_100.ivec","M_128_100.ivec","N_128_100.ivec","E_128_100.ivec"]
#emotion_list = ["A_64.ivec","M_64.ivec","N_64.ivec","E_64.ivec"]

def cosine_distance(a, b):
	if len(a) != len(b):
		raise ValueError, "a and b must be same length"
	numerator = sum(tup[0] * tup[1] for tup in izip(a,b))
	denoma = sum(avalue ** 2 for avalue in a)
	denomb = sum(bvalue ** 2 for bvalue in b)
	result = numerator / (numpy.sqrt(denoma)*numpy.sqrt(denomb))
	return result

def cosine_score(client_ivectors, probe_ivector):
	"""Computes the score for the given model and the given probe using the scoring function"""
	scores = []
	for ivec in client_ivectors:
		scores.append(cosine_distance(ivec, probe_ivector))
	return numpy.max(scores)

results = []
accuracy_list = []
# cosine_score
for probe in emotion_list:
	correct = 0.0
	wrong = 0.0
	probe_ivectors_path = os.path.join(test_ivectors_path, probe)
	print 'Probe Emotion:', probe
	probe_ivec = bob.io.HDF5File(probe_ivectors_path)
	probe_ivectors = probe_ivec.read('ivec')
	probe_ivectors = numpy.array(probe_ivectors)
	for probe_ivector in probe_ivectors:
		probe_result = []
		for model in emotion_list:
			#print probe.split('.',1)[0], ' vs. ', model.split('.',1)[0]
			model_ivec_path = os.path.join(train_ivectors_path, model)
			model_ivec = bob.io.HDF5File(model_ivec_path)
			model_ivectors = model_ivec.read('ivec')
			model_ivectors = numpy.array(model_ivectors)
			score = cosine_score(model_ivectors, probe_ivector)
			probe_result += [(probe.split('.',1)[0], model.split('.',1)[0], score)]
			#results += [(probe.split('.',1)[0], model.split('.',1)[0], score)]
		s_max = 0
		predict = []
		for item in probe_result:
			 p, m, s = item
			 print p, m, s
			 if s > s_max:
			 	s_max = s
			 	predict = item
		print predict
		p, m, s = predict
		if p == m:
			correct = correct + 1
		else:
			wrong = wrong + 1
	accuracy = correct / ( correct + wrong)
	accuracy_list += [(probe.split('.',1)[0], accuracy)]

df = pandas.DataFrame(accuracy_list)
df.columns = ["probe", "accuracy"]
df.to_csv("results/ivector_cosine_accuracy_256G_39D_100.csv")
