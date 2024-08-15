#!/bin/bash 

# Here is how to use the script. 
# This script expects a single argument, which is the name of the benchmark 
# dataset you want to build with. 
# For example, to build with the 'sift-128-euclidean' dataset, you would run:
# ./tools/run_build.sh sift-128-euclidean
# These benchmarks can be found here: 
# Reference: https://github.com/erikbern/ann-benchmarks
# The datasets will be expected to be under the 'data' directory. 
# 
# One common thing to do is to first run 
# ./bin/download_anns_datasets.sh sift-128-euclidean 
# to download the dataset. This will fetch the dataset and put it in the
# 'data' directory.
# The directory structure should look like this:
# data/
#   sift-128-euclidean/
#     sift-128-euclidean.train.npy
#     sift-128-euclidean.query.npy
#     sift-128-euclidean.test.npy 
#
# NOTE: This is really just a convenience script for quickly testing the 
# C++ code. If you want more parameter configuration, you should look at 
# the 'construct_npy.cpp' file. Here we just hardcode the parameters. 

# Make sure we're at the top level directory.
cd "$(dirname "$0")/.."


# Check if the user provided a dataset name.
if [ -z "$1" ]
    then 
        echo "No dataset name provided. Please provide a dataset name."
        exit 1
fi

# Check if the dataset exists.
if [ ! -d "data/$1" ]
    then 
        echo "Dataset $1 does not exist. Please download the dataset first."
        exit 1
fi

# Now get the data
echo "Building with dataset $1"
DATASET_NAME=$1
TRAIN_DATASET_PATH="data/$DATASET_NAME/$DATASET_NAME.train.npy"

# Check if the training dataset exists.
if [ ! -f $TRAIN_DATASET_PATH ]
    then 
        echo "Training dataset $TRAIN_DATASET_PATH does not exist. Please download the dataset first."
        exit 1
fi


# If the dataset has 'euclidean' in the name, then we use L2 distance. Otherwise,
# we use inner product.
if [[ $DATASET_NAME == *"euclidean"* ]]
    then 
        METRIC=0
    else
        METRIC=1
fi

# Summary comments from construct_npy.cpp for reference
#    std::clog << "Usage: " << std::endl;
#    std::clog << "construct <quantize> <metric> <data> <M> <ef_construction> "
#                 "<build_num_threads> <outfile>"
#              << std::endl;
#    std::clog << "\t <quantize> int, 0 for no quantization, 1 for quantization"
#              << std::endl;
#    std::clog << "\t <metric> int, 0 for L2, 1 for inner product (angular)"
#              << std::endl;
#    std::clog << "\t <data> npy file from ann-benchmarks" << std::endl;
#    std::clog << "\t <M>: int " << std::endl;
#    std::clog << "\t <ef_construction>: int " << std::endl;
#    std::clog << "\t <build_num_threads>: int " << std::endl;
#    std::clog << "\t <outfile>: where to stash the index" << std::endl;
# 


./build/construct_npy \
    0 \
    $METRIC \
    $TRAIN_DATASET_PATH \
    32 \
    100 \
    16 \
    "data/$DATASET_NAME/$DATASET_NAME.index"




 
