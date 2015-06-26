#!/bin/sh
# Script which computes the gmmubm score for semaine datsets with varying number of gaussians and number of test folds into consideration.

echo "Starting 64G Gaussians Experiments ... "
#nohup python gmmubm_score_semaine.py 64 25 > logs/gmm_score_semaine_25fold_64G.log &
#nohup python gmmubm_score_semaine.py 64 100 > logs/gmm_score_semaine_100fold_64G.log &
#nohup python gmmubm_score_semaine.py 64 250 > logs/gmm_score_semaine_250fold_64G.log &
nohup python gmmubm_score_semaine.py 64 500 > logs/gmm_score_semaine_500fold_64G.log &

echo "Starting 128G Gaussians Experiments ... "
#nohup python gmmubm_score_semaine.py 128 25 > logs/gmm_score_semaine_25fold_128G.log &
#nohup python gmmubm_score_semaine.py 128 100 > logs/gmm_score_semaine_100fold_128G.log &
#nohup python gmmubm_score_semaine.py 128 250 > logs/gmm_score_semaine_250fold_128G.log &
nohup python gmmubm_score_semaine.py 128 500 > logs/gmm_score_semaine_500fold_128G.log &

wait

echo "Starting 256G Gaussians Experiments ... "
#nohup python gmmubm_score_semaine.py 256 25 > logs/gmm_score_semaine_25fold_256G.log &
#nohup python gmmubm_score_semaine.py 256 100 > logs/gmm_score_semaine_100fold_256G.log &
#nohup python gmmubm_score_semaine.py 256 250 > logs/gmm_score_semaine_250fold_256G.log &
nohup python gmmubm_score_semaine.py 256 500 > logs/gmm_score_semaine_500fold_256G.log &

echo "Starting 512G Gaussians Experiments ... "
#nohup python gmmubm_score_semaine.py 512 25 > logs/gmm_score_semaine_25fold_512G.log &
#nohup python gmmubm_score_semaine.py 512 100 > logs/gmm_score_semaine_100fold_512G.log &
#nohup python gmmubm_score_semaine.py 512 250 > logs/gmm_score_semaine_250fold_512G.log &
nohup python gmmubm_score_semaine.py 512 500 > logs/gmm_score_semaine_500fold_512G.log &

echo "Experiments Finshed!"