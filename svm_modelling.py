'''
SVM - Support Vector Machines Modelling to detect voice activity on Semaine Data (This data will be later used for training and testing based on GMM's and ivectors with other preprocessing methods employed)
@uthor: Ankit Bahuguna
'''

from sklearn import svm
from sklearn import cross_validation
from sklearn import preprocessing
import numpy as np
import time
import sys
import cPickle as pickle
import scipy.sparse
from sklearn.externals import joblib
import os

if len(sys.argv)!=5:
    print '\nUsage: python svm_modelling.py <speech_vector_file> <class_label_file> <test_data_dir> <test_data_output_dir>'
    sys.exit()

speech_vector_file = sys.argv[1]
class_label_file = sys.argv[2]
test_data_dir = sys.argv[3]
test_data_dir_output = sys.argv[4]

if not os.path.exists(test_data_dir_output):
    os.makedirs(test_data_dir_output)


def unpackMfccVector(noise_mix_speech_file):
    with open(noise_mix_speech_file, 'rb') as infile1:
        mfcc = pickle.load(infile1)
    infile1.close()
    mfcc = scipy.sparse.coo_matrix((mfcc), dtype=np.float64).toarray()
    return mfcc

def saveVectorToDisk(mfcc_vector_output_file, speech_vector_final):
    mfcc_vector_file = open(mfcc_vector_output_file, 'w')
    #temp1 = scipy.sparse.coo_matrix(speech_vector_final)
    pickle.dump(speech_vector_final,mfcc_vector_file,-1)
    mfcc_vector_file.close()


def main():
    #We have chosen l2 normalization to normalize the mfcc speech vector over the entire set of frames.
    #Training Data -- Speech Vector File
    with open(speech_vector_file, 'rb') as infile1:
       InputData = pickle.load(infile1)
    InputDataSpeech = preprocessing.normalize(InputData,norm='l2')
    infile1.close()

    # Target Values -- Class Label Files.
    with open(class_label_file, 'rb') as infile2:
       TargetData = pickle.load(infile2)
    #TargetClassLabelTemp = preprocessing.normalize(TargetData,norm='l2')
    infile2.close()

    print InputDataSpeech.shape
    print TargetData.shape

    TargetClassLabelTemp = np.array(scipy.sparse.coo_matrix((TargetData),dtype=np.float64).toarray()).tolist()
    TargetClassLabel = [int(item) for sublist in TargetClassLabelTemp for item in sublist]


    TargetClassLabel = map(str, TargetClassLabel)
    #print TargetClassLabel

    #Recording the start time.
    start = time.time()
    print 'Initializing Liblinear based SVM Machine'
    #Choosing SVM as our machine learning model.
    #clf_model = svm.LinearSVC(C=1000.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='hinge', multi_class='ovr', penalty='l2',random_state=None, tol=0.0001, verbose=0)

    # Default
    clf_model = svm.LinearSVC()

    print 'Fitting Model to Data'
    # fit() the model to the data
    #clf = clf_model.fit(InputDataSpeech,TargetClassLabel)
    #Save the learned machine model fitted to data (input and target) to disk
    #joblib.dump(clf, 'svmsimple_fitted_model.pkl')

    #Load Model From Memory
    clf = joblib.load('svm_fitted_model.pkl')

    for root, dirs, files in os.walk(test_data_dir):
        path = root.split('/')
        for file in files:
            mfcc_file = os.path.join(test_data_dir, file)
            print 'Predicting on File: '+str(mfcc_file)
            TestData = preprocessing.normalize(unpackMfccVector(mfcc_file),norm='l2')
            out = clf.predict(TestData)
            print out

            name  = file.replace('.dat','')
            outputfile = os.path.join(test_data_dir_output,str(name)+'_predicted.dat')
            saveVectorToDisk(outputfile,out)
            print str(file)+' : VAD Prediction Done!'

    #scores = cross_validation.cross_val_score(clf_model, InputDataSpeech, TargetClassLabel, cv=10)
    #print "\nFinal Accuracy Score: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std()*2)

    #Recording the end time.
    end = time.time()


    print "Total execution time in minutes :: >>"
    print (end - start)/60

    print 'Task is Finished!'

if __name__=="__main__":
    main()