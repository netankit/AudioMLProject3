#!/bin/sh
# Shell Script to Run the KMeansUBM.py for the GiantUBM 39D Dataset with Gaussians 64,128,256 and 512

echo "Starting 64G Training ...  "
nohup python KmeansUBM.py 64 /mnt/alderaan/mlteam3/Assignment3/data/giantubm_mfcc/mfcc_vaded/final_mfcc_39D.dat kmeans_giantubm_39D_64G.hdf5 gmm_giantubm_39D_64G.hdf5 > logs/gmm_giantubm_39D_64G.log &

echo "Starting 128G Training ...  "

nohup python KmeansUBM.py 128 /mnt/alderaan/mlteam3/Assignment3/data/giantubm_mfcc/mfcc_vaded/final_mfcc_39D.dat kmeans_giantubm_39D_128G.hdf5 gmm_giantubm_39D_128G.hdf5 > logs/gmm_giantubm_39D_128G.log &

echo "Starting 256G Training ...  "

nohup python KmeansUBM.py 256 /mnt/alderaan/mlteam3/Assignment3/data/giantubm_mfcc/mfcc_vaded/final_mfcc_39D.dat kmeans_giantubm_39D_256G.hdf5 gmm_giantubm_39D_256G.hdf5 > logs/gmm_giantubm_39D_256G.log &
echo "Starting 512G Training ...  "

nohup python KmeansUBM.py 512 /mnt/alderaan/mlteam3/Assignment3/data/giantubm_mfcc/mfcc_vaded/final_mfcc_39D.dat kmeans_giantubm_39D_512G.hdf5 gmm_giantubm_39D_512G.hdf5 > logs/gmm_giantubm_39D_512G.log &

echo "Finished"

