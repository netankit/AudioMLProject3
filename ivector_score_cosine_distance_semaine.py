import os
import bob
import numpy
import math
from itertools import *
import pandas
import sys


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


def filtr(files, filetype):
    """Filters a file list by the given filename ending to be accepted"""
    return filter(lambda d: 1 if d.endswith(filetype) else 0, files)

num_folds_list = [25,50,100, 250]
num_gaussian_list = [64,128,256,512]

for num_folds in num_folds_list:
	print "Number of Folds: "+str(num_folds)

	for num_gaussian in num_gaussian_list:
		print "Number of Gaussians: "+str(num_gaussian)
		# INPUT DIRECTORY
		ubm_ivectors_path  = '/mnt/alderaan/mlteam3/Assignment3/ivectors_100/39D/128/ubm_noisy_mfcc_39D'

		train_ivectors_path = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/ivectors_100_emotionwise/train/'+str(num_gaussian)+'/'
		test_ivectors_path = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data_copy/test_multiple_folds_ivectors/fold'+str(num_folds)+'/'+str(num_gaussian)+'/'

		emotion_list = ["angry_"+str(num_gaussian)+"_100.ivec","sad_"+str(num_gaussian)+"_100.ivec","neutral_"+str(num_gaussian)+"_100.ivec","relaxed_"+str(num_gaussian)+"_100.ivec","happy_"+str(num_gaussian)+"_100.ivec"]
		#print emotion_list
		#
		emotion_name_list = ["angry", "sad", "happy", "relaxed", "neutral"]




		accuracy_list = []
		# Cosine Distance Scoring for IVectors

		for emotion_original in emotion_name_list:
			correct = 0.0
			wrong = 0.0
			root = test_ivectors_path+emotion_original+'/'
			#print root
			files = filtr(os.listdir(root),'.ivec')
			#print files
			#sys.exit()
			for file in files:
				#emotion_original = file.split("_",1)[0]
				probe_ivectors_path = os.path.join(root,file)
				probe_ivec = bob.io.HDF5File(probe_ivectors_path)
				#print type(probe_ivec)
				probe_ivectors_temp = probe_ivec.read('ivec')
				#print probe_ivectors_temp.shape
				probe_ivectors = numpy.array(probe_ivectors_temp)
				#print probe_ivectors
				#print 'TEST PROBE: ' +str(emotion_original)
				#print "Test Probe Shape: "+str(probe_ivectors.shape)
				#print probe_ivectors
				#sys.exit()

				for probe_ivector in probe_ivectors:
					probe_result = []
					probe_score_final = 0.0
					probe_predicted_emotion = ''
					for model in emotion_list:
						model_name = model.replace("_"+str(num_gaussian)+"_100.ivec", "")
						#print "Trained Model IVEC: "+str(model_name)
						model_ivec_path = os.path.join(train_ivectors_path, model)
						model_ivec = bob.io.HDF5File(model_ivec_path)
						model_ivectors = model_ivec.read('ivec')
						model_ivectors = numpy.array(model_ivectors)
						score = cosine_score(model_ivectors, probe_ivector)
						#print "Cosine Score: "+str(score)
						if probe_score_final < score:
							probe_score_final = score
							probe_predicted_emotion = model_name
					if str(probe_predicted_emotion) == str(emotion_original):
						correct +=1
					else:
						wrong += 1
			#print "# Correct: "+str(correct)
			#print '# Incorrect: '+str(wrong)
			accuracy = correct / (correct + wrong)
			print 'Accuracy for emotion '+str(emotion_original)+' is : '+str(accuracy)
			accuracy_list += [(emotion_original, accuracy)]

		df = pandas.DataFrame(accuracy_list)
		df.columns = ["probe", "accuracy"]
		df.to_csv("results/semaine_results/semaine_ivectors_" + str(num_gaussian)+"G_"+str(num_folds)+"_39D.csv")
print "Finished!"