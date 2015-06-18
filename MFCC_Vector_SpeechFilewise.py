import scipy.io.wavfile as wav
import numpy as np
import math
from features import mfcc
import sys
import os
import cPickle
import scipy.sparse

if len(sys.argv)!=4:
    print '\nUsage: MFCC_Vector_SpeechFilewise.py <noise_mix_speech_dir> <mfcc_vector_output_directory> <mfcc_dimensions>'
    sys.exit()

audio_files_dir= sys.argv[1]
mfcc_vector_output_filedir=sys.argv[2]
mfcc_dimensions = int(sys.argv[3])

if not os.path.exists(mfcc_vector_output_filedir):
	os.makedirs(mfcc_vector_output_filedir)


def getMfccVector(noise_mix_speech_file):
	(rate, signal) = wav.read(noise_mix_speech_file)
	mfcc_vec = mfcc(signal,rate,winlen=0.025,winstep=0.01,numcep=mfcc_dimensions,
          nfilt=mfcc_dimensions*2,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          ceplifter=22,appendEnergy=True)
	return mfcc_vec

def getDataset(audio_files_dir):

	directoryCount = 1
	for root, dirs, files in os.walk(audio_files_dir):
		print "Directory Count: "+str(directoryCount)
		path = root.split('/')
		for file in files:
			speech_vector_final = np.zeros((1,mfcc_dimensions))
			speech_vector_final = np.delete(speech_vector_final, (0), axis=0)

			if (file.lower().endswith('.wav')):
				speechFilePath = os.path.join(root,str(file))
				tmp = os.path.dirname(speechFilePath)
				root_dir_name = os.path.basename(tmp)


				audio_file_fullpath = os.path.join(audio_files_dir,root_dir_name,file)

				mfcc_vector_filename= str('mfcc_'+os.path.splitext(file)[0])+'.dat'


				mfcc_vector_output_filedir_final = os.path.join(mfcc_vector_output_filedir, root_dir_name)


				if not os.path.exists(mfcc_vector_output_filedir_final):
					os.makedirs(mfcc_vector_output_filedir_final)


				mfcc_vector_output_file = os.path.join(mfcc_vector_output_filedir_final,mfcc_vector_filename)



				print "Audio file:"+ audio_file_fullpath

				mfcc_vector =  getMfccVector(audio_file_fullpath)


				speech_vector_final = np.vstack((speech_vector_final,mfcc_vector))

				#Write the mfcc speech vector for speech file.
				mfcc_vector_file = open(mfcc_vector_output_file, 'w')
				temp1 = scipy.sparse.coo_matrix(speech_vector_final)
				cPickle.dump(temp1,mfcc_vector_file,-1)
				mfcc_vector_file.close()

				print "Final Shapes:"
				print "Speech Vector:"+str(speech_vector_final.shape)


		directoryCount = directoryCount+1



#Main Routine
print "Start of the Program ....."

getDataset(audio_files_dir)

print "Program completed Successfully"


