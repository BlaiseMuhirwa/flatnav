#!/bin/bash

# If filename called `sift_128.index` exists, delete it first
if [ -f mnist_784.index ]; then
    rm mnist_784.index
fi

# if [ -f sift_128.index ]; then
#     rm sift_128.index
# fi

# if [ -f glove_25.index ]; then
#     rm glove_25.index
# fi

# if [ -f gist_960.index ]; then
#     rm gist_960.index
# fi

# if [ -f deep1b_96.index ]; then
#     rm deep1b_96.index
# fi

# Build the index for MNIST 
build/construct_npy 0 0 data/mnist-784-euclidean/mnist-784-euclidean.train.npy 16 128 mnist_784.index

# # Query MNIST
build/query_npy 0 mnist_784.index data/mnist-784-euclidean/mnist-784-euclidean.test.npy data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy 256 100 0 0

# # Query MNIST with reordering
# build/query_npy 0 mnist_784.index data/mnist/mnist-784-euclidean.test.npy data/mnist/mnist-784-euclidean.gtruth.npy 256,512 100 1

# Build the index
# build/construct_npy 1 0 data/sift/sift-128-euclidean.train.npy 16 128 sift_128.index

# # Query 
# build/query_npy 0 sift_128.index data/sift/sift-128-euclidean.test.npy data/sift/sift-128-euclidean.gtruth.npy 256,512 100 0

# Build the index for GloVe
# build/construct_npy 1 1 data/glove/glove-25-angular.train.npy 16 128 glove_25.index

# Query GloVe without reordering 
# build/query_npy 1 glove_25.index data/glove/glove-25-angular.test.npy data/glove/glove-25-angular.gtruth.npy 128,256 100 0 1

# # Query GloVe with reordering
# build/query_npy 1 glove_25.index data/glove/glove-25-angular.test.npy data/glove/glove-25-angular.gtruth.npy 256,512 100 1

# Build the index for GIST
# build/construct_npy 0 0 data/gist/gist-960-euclidean.train.npy 32 128 gist_960.index

# # Query GIST without reordering
# build/query_npy 0 gist_960.index data/gist/gist-960-euclidean.test.npy data/gist/gist-960-euclidean.gtruth.npy 128,256 100 0

# # Query GIST with reordering
# echo "querying with re-ordering \n"
# build/query_npy 0 gist_960.index data/gist/gist-960-euclidean.test.npy data/gist/gist-960-euclidean.gtruth.npy 128,256 100 1


# Build the index for DEEP1B
# build/construct_npy 0 1 data/deep1b/deep-image-96-angular.train.npy 32 128 deep1b_96.index

# # Query DEEP1B without reordering
# build/query_npy 1 deep1b_96.index data/deep1b/deep-image-96-angular.test.npy data/deep1b/deep-image-96-angular.gtruth.npy 128,256 100 0

# # Query DEEP1B with reordering
# build/query_npy 1 deep1b_96.index data/deep1b/deep-image-96-angular.test.npy data/deep1b/deep-image-96-angular.gtruth.npy 128,256 100 1