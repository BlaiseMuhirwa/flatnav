#!/bin/bash

# If filename called `sift_128.index` exists, delete it first
if [ -f mnist_784.index ]; then
    rm mnist_784.index
fi

# Build the index for MNIST 
build/construct_npy 1 0 data/mnist/mnist-784-euclidean.train.npy 16 128 mnist_784.index

# Query MNIST
build/query_npy 0 mnist_784.index data/mnist/mnist-784-euclidean.test.npy data/mnist/mnist-784-euclidean.gtruth.npy 128,256 100 0

# # Build the index
# build/construct_npy 1 0 data/sift/sift-128-euclidean.train.npy 16 128 sift_128.index

# # Query 
# build/query_npy 0 sift_128.index data/sift/sift-128-euclidean.test.npy data/sift/sift-128-euclidean.gtruth.npy 128,256 100 0




