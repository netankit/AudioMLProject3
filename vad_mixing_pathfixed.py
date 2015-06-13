'''
vad_mixing.py: This is a modified version of original vad_mixing.py
Original Author: Florian Schulze
Modified by: Ankit Bahuguna

'''

import scipy.io.wavfile
import scipy.signal
import os
import glob
import math
import random
import numpy as np
from joblib import Parallel, delayed
import audiotools
import time
import shutil
import sys

if len(sys.argv)!=3:
	 print '\nUsage: python vad_mixing.py <speaker_speech_dir> <output_directory_path>\n'
	 sys.exit()

impRespPath = '/mnt/tatooine/data/impulse_responses/16kHz/wavs64b'
noisePath = '/mnt/tatooine/data/noise/noise_equal_concat/train'
rootPathCleanCorpus = sys.argv[1]
#rootPathCleanCorpus = '/mnt/tatooine/data/vad_speaker_recog/TIMIT_Buckeye/vad' # MAIN
#rootPathCleanCorpus = '/mnt/alderaan/mlteam3/data/sample' #TESTING ONLY

replay_gain_ref_path = '/mnt/tatooine/data/ref_pink_16k.wav'

#outputPath1 = '/mnt/alderaan/mlteam3/data/assignment2data' # MAIN
#outputPath1 = '/mnt/alderaan/mlteam3/data/outputsample' #TESTING ONLY
outputPath1 = sys.argv[2]

#pobability of pocket movement
probOfPocketMov = 0.2
#pobabiliy of noise being added up to the clean file
probOfNoise = 0.8

#max size of break in sec
maxLenSilence = 0.2;
#min size of break in sec
minLenSilence = 0;

numJobs = 3

wantedFs = 16000
#maximum of the SNR in db (snrMaxDecimal = 10)
snrMax = 20
#minimum of the SNR in db (snrMinDecimal = 0.79433)
snrMin = -2

#Define indices for picking impulse responses
irPhoneIndex = 1
irTestPosIndex = 1

###
# End of config
###

noises = []
silence = []
pocketNoise = []
impResps = []

def cacheNoiseFiles():
	wavFileList = glob.glob(os.path.join(noisePath, '*.wav'))
	for wavFile in wavFileList:
		(fs, samples) = scipy.io.wavfile.read(wavFile)
		#print "Noise SAMPLES (dtype): "+str(samples.dtype)
		noises.append(samples)
		print 'Noise file %s read.' % (wavFile)

	print 'Noise cached in memory.'


def cacheImpulseResponses():
	for root, dirs, files in os.walk(impRespPath):
		path = root.split('/')
		for file in files:
			irSignals = []
			if(file.lower().endswith('.wav')):
				(_, samples) = scipy.io.wavfile.read(os.path.join(impRespPath, file))
				#print "IR SAMPLES (dtype): "+str(samples.dtype)
				irSignals.append(samples)
			impResps.append(irSignals)
	print 'IRs cached in memory.'



def getRandomFadedNoise(nSamples):
	while(True):
		index = random.randint(0, len(noises)-1)
		if len(noises[index] >= nSamples):
			#this noise file is long enough, use it
			break

	noiseSignal = noises[index]

	#get random start point
	rangePotStartPoints = len(noiseSignal) - nSamples
	startInd = math.ceil(random.random() * rangePotStartPoints)
	noiseSegment = noiseSignal[startInd:startInd+nSamples]

	#fade to avoid artifacts
	fadeLength = min(len(noiseSegment), 2000)/2
	noiseSegment[:fadeLength] *= np.linspace(0, 1, num=fadeLength)
	noiseSegment[-fadeLength:] *= np.linspace(1, 0, num=fadeLength)

	return noiseSegment

def mixFilesInSpeakerPath(spInd, folder):
    speakerPath = os.path.join(rootPathCleanCorpus, folder)
    wavFileList = glob.glob(os.path.join(speakerPath, '*.wav'))
    print 'Starting speaker %s...' % (folder)
    for (ind,wavFile) in enumerate(wavFileList):

        try:
            (fs, samples) = scipy.io.wavfile.read(wavFile)
            samples = samples.astype(np.float64)/65536.0
            #print 'Speech snippet %s read.' % (wavFile)
            tmp_file = os.path.dirname(wavFile)
            root_dir_name = os.path.basename(tmp_file)
            root_filename = str(os.path.splitext(os.path.basename(wavFile))[0])+'_out'+str('.wav')
            print '*** Root Name file: '+str(root_filename)

            #read annotation
            #with open(wavFile.replace("wav", "ano")) as f:
            #	anoList = [int(line.rstrip()) for line in list(f)]

            #if len(anoList) != len(samples):
            #	print 'Mismatch in size between annotation and track!'

            #get replaygain stats of current file
            file_rplgain = list(audiotools.calculate_replay_gain([ \
                audiotools.open(wavFile) \
                ]))[0][1]

            #calculate gain to ref file and normalize accordingly
            gain = file_rplgain - ref_rplgain
            normSignal = samples * (10**(gain/20.0))

            if (random.random() < probOfNoise):
                #mix with noise of same size
                noise = getRandomFadedNoise(len(normSignal))
                #calculate the random SNR
                randomSNR = snrMin + (snrMax-snrMin) * random.random()
                #amplify signal by reducing noise
                noise /= 10**(randomSNR/20) #normSignal *= 10**(randomSNR/20);
                normSignal += noise
            # CONVOLVING NOISE MIXED SPEECH SIGNALS WITH THE IMPSULSE RESPONSE SIGNALS
            irTrain1 = random.choice(impResps)
            irTrain2 = np.asarray(irTrain1)
            irTrain = irTrain2.flatten()

            #print "irTrain Type: "+str(type(irTrain))
            #print "irTrain Value: "+str(irTrain)
            #print "irTrain Length: "+str(len(irTrain))
            #print "irTrain Shape: "+str(irTrain.shape)
            #print "normSignal Length: "+str(len(normSignal))
            convolvedSignal1 = scipy.signal.fftconvolve(normSignal, irTrain)[:len(normSignal)]
            outputDir = os.path.join(outputPath1, root_dir_name)
            #print '*** Output DIR: '+str(outputDir)
            #print '*** Root Name file: '+str(root_filename)
            outputFile1 = os.path.join(outputDir,  root_filename)
            #print '*** Output File Name: '+str(outputFile1)
            if not os.path.exists(outputDir):
                os.makedirs(outputDir)
		#shutil.copyfile(wavFile.replace("wav", "ano"), outputFile1.replace("wav", "ano"))
		#print 'Writing %s.' % (outputFile)
            scipy.io.wavfile.write(outputFile1, wantedFs, convolvedSignal1)
		#f = open(outputFile1.replace("wav", "ano"),'w')
		#for (ind,line) in enumerate(anoList):
		#	if ind == (len(anoList) - 1):
		#		#no \n at end of file
		#		f.write("%i" % (line))
		#	else:
		#		f.write("%i\n" % (line))
		#f.close()
        except ValueError:
            print "Value Error"
    print 'Speaker %s done' % (folder)



if __name__ == '__main__':
	cacheNoiseFiles()
	cacheImpulseResponses()

	#replaygain val of reference file
	ref_rplgain = list(audiotools.calculate_replay_gain([ \
		audiotools.open(replay_gain_ref_path) \
		]))[0][1]

	#get folder names (folders = speakers)
	all_speaker_names = os.walk(rootPathCleanCorpus).next()[1]
	print '%d speakers detected.' % (len(all_speaker_names))

	#USING SINGLE PROCESS
	for (ind,speaker) in enumerate(all_speaker_names):
		mixFilesInSpeakerPath(ind,speaker)

	# UTILIZING MULTIPLE PROCESSES via joblib.
	#results = Parallel(n_jobs=numJobs)(delayed(mixFilesInSpeakerPath)(ind,speaker) \
	#	for (ind,speaker) in enumerate(all_speaker_names))
	print 'All done.'