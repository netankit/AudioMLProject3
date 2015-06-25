#!/bin/sh
# Shell Script to Run the KMeansUBM_Emotion.py for the semaine emotion [angry, happy, neutral, relaxed sad ] 39D mfcc Dataset with Gaussians 64,128,256 and 512
#@uthor: Ankit Bahuguna

echo "Starting Angry ... "
echo "Starting 64G Training ...  "
nohup python KmeansUBM_Emotion.py 64 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/angry_train.dat kmeans_angry_gmm_semaine_39D_64G.hdf5 gmm_angry_gmm_semaine_39D_64G.hdf5 > logs/angry_gmm_semaine_39D_64G.log &

echo "Starting 128G Training ...  "
nohup python KmeansUBM_Emotion.py 128 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/angry_train.dat kmeans_angry_gmm_semaine_39D_128G.hdf5 gmm_angry_gmm_semaine_39D_128G.hdf5 > logs/angry_gmm_semaine_39D_128G.log &

echo "Starting 256G Training ...  "
nohup python KmeansUBM_Emotion.py 256 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/angry_train.dat kmeans_angry_gmm_semaine_39D_256G.hdf5 gmm_angry_gmm_semaine_39D_256G.hdf5 > logs/angry_gmm_semaine_39D_256G.log &

echo "Starting 512G Training ...  "
nohup python KmeansUBM_Emotion.py 512 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/angry_train.dat kmeans_angry_gmm_semaine_39D_512G.hdf5 gmm_angry_gmm_semaine_39D_512G.hdf5 > logs/angry_gmm_semaine_39D_512G.log &



echo "Starting Happy ... "
echo "Starting 64G Training ...  "
nohup python KmeansUBM_Emotion.py 64 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/happy_train.dat kmeans_happy_gmm_semaine_39D_64G.hdf5 gmm_happy_gmm_semaine_39D_64G.hdf5 > logs/happy_gmm_semaine_39D_64G.log &

echo "Starting 128G Training ...  "
nohup python KmeansUBM_Emotion.py 128 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/happy_train.dat kmeans_happy_gmm_semaine_39D_128G.hdf5 gmm_happy_gmm_semaine_39D_128G.hdf5 > logs/happy_gmm_semaine_39D_128G.log &

wait

echo "Starting 256G Training ...  "
nohup python KmeansUBM_Emotion.py 256 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/happy_train.dat kmeans_happy_gmm_semaine_39D_256G.hdf5 gmm_happy_gmm_semaine_39D_256G.hdf5 > logs/happy_gmm_semaine_39D_256G.log &

echo "Starting 512G Training ...  "
nohup python KmeansUBM_Emotion.py 512 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/happy_train.dat kmeans_happy_gmm_semaine_39D_512G.hdf5 gmm_happy_gmm_semaine_39D_512G.hdf5 > logs/happy_gmm_semaine_39D_512G.log &



echo "Starting Neutral ... "
echo "Starting 64G Training ...  "
nohup python KmeansUBM_Emotion.py 64 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/neutral_train.dat kmeans_neutral_gmm_semaine_39D_64G.hdf5 gmm_neutral_gmm_semaine_39D_64G.hdf5 > logs/neutral_gmm_semaine_39D_64G.log &

echo "Starting 128G Training ...  "
nohup python KmeansUBM_Emotion.py 128 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/neutral_train.dat kmeans_neutral_gmm_semaine_39D_128G.hdf5 gmm_neutral_gmm_semaine_39D_128G.hdf5 > logs/neutral_gmm_semaine_39D_128G.log &

echo "Starting 256G Training ...  "
nohup python KmeansUBM_Emotion.py 256 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/neutral_train.dat kmeans_neutral_gmm_semaine_39D_256G.hdf5 gmm_neutral_gmm_semaine_39D_256G.hdf5 > logs/neutral_gmm_semaine_39D_256G.log &

echo "Starting 512G Training ...  "
nohup python KmeansUBM_Emotion.py 512 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/neutral_train.dat kmeans_neutral_gmm_semaine_39D_512G.hdf5 gmm_neutral_gmm_semaine_39D_512G.hdf5 > logs/neutral_gmm_semaine_39D_512G.log &

wait

echo "Starting Relaxed ... "
echo "Starting 64G Training ...  "
nohup python KmeansUBM_Emotion.py 64 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/relaxed_train.dat kmeans_relaxed_gmm_semaine_39D_64G.hdf5 gmm_relaxed_gmm_semaine_39D_64G.hdf5 > logs/relaxed_gmm_semaine_39D_64G.log &

echo "Starting 128G Training ...  "
nohup python KmeansUBM_Emotion.py 128 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/relaxed_train.dat kmeans_relaxed_gmm_semaine_39D_128G.hdf5 gmm_relaxed_gmm_semaine_39D_128G.hdf5 > logs/relaxed_gmm_semaine_39D_128G.log &

echo "Starting 256G Training ...  "
nohup python KmeansUBM_Emotion.py 256 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/relaxed_train.dat kmeans_relaxed_gmm_semaine_39D_256G.hdf5 gmm_relaxed_gmm_semaine_39D_256G.hdf5 > logs/relaxed_gmm_semaine_39D_256G.log &

echo "Starting 512G Training ...  "
nohup python KmeansUBM_Emotion.py 512 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/relaxed_train.dat kmeans_relaxed_gmm_semaine_39D_512G.hdf5 gmm_relaxed_gmm_semaine_39D_512G.hdf5 > logs/relaxed_gmm_semaine_39D_512G.log &



echo "Starting Sad ... "
echo "Starting 64G Training ...  "
nohup python KmeansUBM_Emotion.py 64 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/sad_train.dat kmeans_sad_gmm_semaine_39D_64G.hdf5 gmm_sad_gmm_semaine_39D_64G.hdf5 > logs/sad_gmm_semaine_39D_64G.log &

echo "Starting 128G Training ...  "
nohup python KmeansUBM_Emotion.py 128 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/sad_train.dat kmeans_sad_gmm_semaine_39D_128G.hdf5 gmm_sad_gmm_semaine_39D_128G.hdf5 > logs/sad_gmm_semaine_39D_128G.log &

wait

echo "Starting 256G Training ...  "
nohup python KmeansUBM_Emotion.py 256 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/sad_train.dat kmeans_sad_gmm_semaine_39D_256G.hdf5 gmm_sad_gmm_semaine_39D_256G.hdf5 > logs/sad_gmm_semaine_39D_256G.log &

echo "Starting 512G Training ...  "
nohup python KmeansUBM_Emotion.py 512 /mnt/alderaan/mlteam3/Assignment3/data/Semaine/emotionwise_data/train/sad_train.dat kmeans_sad_gmm_semaine_39D_512G.hdf5 gmm_sad_gmm_semaine_39D_512G.hdf5 > logs/sad_gmm_semaine_39D_512G.log &


echo "Done!"
