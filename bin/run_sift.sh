#!/bin/bash

# If filename called `sift_128.index` exists, delete it first
if [ -f sift_128.index ]; then
    rm sift_128.index
fi

# Build the index
build/construct_npy 1 0 data/sift/sift-128-euclidean.train.npy 16 128 sift_128.index

# Query 
build/query_npy 0 sift_128.index data/sift/sift-128-euclidean.test.npy data/sift/sift-128-euclidean.gtruth.npy 128,256 100 0




