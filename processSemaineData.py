'''
Processes noisy semine dataset "mix.wav files" and adapts anotations based on arousal and valence values.
@uthor: Ankit Bahuguna

'''

import os
import cPickle
import numpy as np
import scipy


mfcc_raw_dir = "/mnt/alderaan/mlteam3/Assignment3/data/Semine/mfcc_raw/39D/"
anno_dir = "/mnt/tatooine/data/emotion/Semaine/Sessions/"

# Output Directories
mfcc_final_out_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/mfcc_final/39D/'
emo_final_out_dir = '/mnt/alderaan/mlteam3/Assignment3/data/Semine/emo_final/39D/'


if not os.path.exists(mfcc_final_out_dir):
    os.makedirs(mfcc_final_out_dir)
if not os.path.exists(emo_final_out_dir):
    os.makedirs(emo_final_out_dir)

# Emotion  Classificaton Labels
emotionCodes_Happy = 1
emotionCodes_Relaxed = 2
emotionCodes_Neutral = 3
emotionCodes_Sad = 4
emotionCodes_Angry = 5


def getSemaineEmotion(valence, arousal):
    emotion=[]
    emo=1
    for i in range(min(len(valence),len(arousal))):
        if abs(valence[i])<0.15 and abs(arousal[i])<0.15:
            emo=emotionCodes_Neutral
        else:
            if valence[i]>0:
                if arousal[i]>0:
                    emo=emotionCodes_Happy
                else:
                    emo=emotionCodes_Relaxed
            else:
                if arousal[i]>0:
                    emo=emotionCodes_Angry
                else:
                    emo=emotionCodes_Sad
        if not isinstance( emo, ( int, long ) ):
            emo=emo.value
        emotion.append(emo)
    return emotion


def iter_load_txt_col(filename, col ,delimiter=' '):
    data = []
    with open(filename, 'r') as infile:
        for line in infile:
            line = line.rstrip().split(delimiter)
            if len(line)>=col+1:
                data.append(float(line[col]))
    return data

def iter_load_txt(filename, delimiter=' '):
    data = []
    with open(filename, 'r') as infile:
        for line in infile:
            line = line.rstrip().split(delimiter)
            data.append(line)
    return data

def filtr(files, filetype):
    """Filters a file list by the given filename ending to be accepted"""
    return filter(lambda d: 1 if d.endswith(filetype) else 0, files)

def unpackMfccVector(mfcc_vector_file):
    with open(mfcc_vector_file, 'rb') as infile1:
        mfcc = cPickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc

def saveVectorToDisk(mfcc_vector_output_file, speech_vector_final):
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    cPickle.dump(temp1,mfcc_vector_file,-1)
    mfcc_vector_file.close()

def main():
    #soundfiles =  os.walk(mfcc_raw_dir).next()[1])
    for subdir, dirs, files in os.walk(mfcc_raw_dir):
        # iterate over all wavs in the subdirectories
        for file in files:
            # Append all data to array with filename as identifier
            name=file.replace('.dat', '')
            name = name.replace('mfcc_', '')
            mfcc_vector_data_output_file = os.path.join(mfcc_final_out_dir, str(name)+'.dat')
            mfcc_vector_emo_output_file = os.path.join(emo_final_out_dir,str(name)+'.dat')

            mfcc_file = os.path.join(mfcc_raw_dir,file)

            print "Current MFCC File Processed: "+str(name)

            # Loads the precomputed  noise mixed speech mfcc vectors from disk for the given file
            frame_mfccs = unpackMfccVector(mfcc_file)

            valencefile = filtr(os.listdir(anno_dir+name+'/'),'DV.txt')
            anotationsv=[]
            for valence in valencefile:
                anotationsv.append(iter_load_txt_col(anno_dir+name+'/'+valence,1)[:len(frame_mfccs)-250])


            arousalfile = filtr(os.listdir(anno_dir+name+'/'),'DA.txt')
            anotationsa=[]
            for arousal in arousalfile:
                anotationsa.append(iter_load_txt_col(anno_dir+name+'/'+arousal,1)[:len(frame_mfccs)-250])

            if len(valencefile)>1 and len(arousalfile)>1:
                meanvalence= np.divide(np.sum(anotationsv,axis=0),len(valencefile))
                meanarousal= np.divide(np.sum(anotationsa,axis=0),len(arousalfile))
                emotion = getSemaineEmotion(meanvalence,meanarousal)
                data_emos = []
                data_mfccs = []
                for i in range(min(len(frame_mfccs),len(meanarousal),len(meanvalence))):
                    data_emos.append(emotion[i])
                    data_mfccs.append(frame_mfccs[i,0:])

            data_emos_arr = np.asarray(data_emos)
            data_mfccs_arr = np.asarray(data_mfccs)
            print 'MFCC SHAPE: '+str(data_mfccs_arr.shape)
            print 'EMO SHAPE: '+str(data_emos_arr.shape)
            saveVectorToDisk(mfcc_vector_data_output_file, data_mfccs_arr)
            print"MFCC File Written!"
            saveVectorToDisk(mfcc_vector_emo_output_file, data_emos_arr)
            print"EMO File Written!"
    print "Finished!"
if __name__  == "__main__":
    main()