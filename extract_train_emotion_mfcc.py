import sys, os, shutil
import argparse
import bob
import numpy as np
import cPickle
import scipy.sparse


'''
# Emotion  Classificaton Labels
emotionCodes_Happy = 1
emotionCodes_Relaxed = 2
emotionCodes_Neutral = 3
emotionCodes_Sad = 4
emotionCodes_Angry = 5
'''

semaine_dir_train = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/concat_train_test_vaded/train/39D/'
semaine_dir_test = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/concat_train_test_vaded/test/39D/'


emotionwise_data_train = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/'
emotionwise_data_test = '/mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/test/'

if not os.path.exists(emotionwise_data_train):
    os.makedirs(emotionwise_data_train)
if not os.path.exists(emotionwise_data_test):
    os.makedirs(emotionwise_data_test)

mfcc_dimensions = 39



def getMfccVector(noise_mix_speech_file):
    (rate, signal) = wav.read(noise_mix_speech_file)
    mfcc_vec = mfcc(signal,rate,winlen=0.025,winstep=0.01,numcep=mfcc_dimensions,
          nfilt=mfcc_dimensions*2,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
          ceplifter=22,appendEnergy=True)
    return mfcc_vec

def unpackMfccVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc

def saveVectorToDisk(mfcc_vector_output_file, speech_vector_final):
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()


def seperateEmotions(semaine_dir, emotionwise_data):
    happy_vec = np.delete(np.zeros((1,mfcc_dimensions)), (0), axis=0)
    relaxed_vec = np.delete(np.zeros((1,mfcc_dimensions)), (0), axis=0)
    neutral_vec = np.delete(np.zeros((1,mfcc_dimensions)), (0), axis=0)
    sad_vec = np.delete(np.zeros((1,mfcc_dimensions)), (0), axis=0)
    angry_vec = np.delete(np.zeros((1,mfcc_dimensions)), (0), axis=0)

    seminefile = os.path.join(semaine_dir, "mfcc_39D.dat")
    emotionfile = os.path.join(semaine_dir,"emotion_39D.dat")

    semaine_vec = unpackMfccVector(seminefile)
    emotion_vec = unpackMfccVector(emotionfile)

    print semaine_vec.shape
    X,Y = semaine_vec.shape
    print emotion_vec.shape


    count = 0
    for row in semaine_vec:
        if (int(emotion_vec[0][count]) == int(1)):
            happy_vec = np.vstack((happy_vec, row))

        if (int(emotion_vec[0][count]) == int(2)):
            relaxed_vec = np.vstack((relaxed_vec,row))

        if (int(emotion_vec[0][count]) == int(3)):
            neutral_vec = np.vstack((neutral_vec,row))

        if (int(emotion_vec[0][count]) == int(4)):
            sad_vec = np.vstack((sad_vec,row))

        if (int(emotion_vec[0][count]) == int(5)):
            angry_vec = np.vstack((angry_vec,row))
        print "ROW's Left:" +str(int(X-count))
        count +=1

    happy_file = os.path.join(emotionwise_data,"happy_train.dat")
    relaxed_file = os.path.join(emotionwise_data,"relaxed_train.dat")
    neutral_file = os.path.join(emotionwise_data,"neutral_train.dat")
    sad_file =  os.path.join(emotionwise_data,"sad_train.dat")
    angry_file = os.path.join(emotionwise_data,"angry_train.dat")

    print "Happy: "+str(happy_vec.shape)
    print "Relaxed: "+str(relaxed_vec.shape)
    print "Neutral: "+str(neutral_vec.shape)
    print "Sad: "+str(sad_vec.shape)
    print "Angry: "+str(angry_vec.shape)

    print "Saving Vectors to Disk ..."
    saveVectorToDisk(happy_file, happy_vec)
    saveVectorToDisk(relaxed_file, relaxed_vec)
    saveVectorToDisk(neutral_file, neutral_vec)
    saveVectorToDisk(sad_file, sad_vec)
    saveVectorToDisk(angry_file, angry_vec)


def main():
    seperateEmotions(semaine_dir_train, emotionwise_data_train)
    seperateEmotions(semaine_dir_test,emotionwise_data_test)

    print "Done!"

if __name__=="__main__":
    main()
