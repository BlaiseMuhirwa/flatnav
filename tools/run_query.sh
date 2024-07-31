#!/bin/bash 

# Here is how to use the script.
# This script expects a single argument, which is the name of the benchmark
# dataset you want to query with.
# For example, to query with the 'sift-128-euclidean' dataset, you would run:
# ./tools/run_query.sh sift-128-euclidean
# The script will assume that you have already built the index with the same
# dataset. If you haven't, you should run the 'run_build.sh' script first.
# The index should be under data/$1/$1.index



# Reference doc from query_npy.cpp
#    std::clog << "Usage: " << std::endl;
#    std::clog << "query <space> <index> <queries> <gtruth> <ef_search> <k> "
#                 "<Reorder ID> <Quantized>"
#              << std::endl;
#    std::clog << "\t <data> <queries> <gtruth>: .npy files (float, float, int) "
#                 "from ann-benchmarks"
#              << std::endl;
#    std::clog << "\t <M>: int number of links" << std::endl;
#    std::clog << "\t <ef_construction>: int " << std::endl;
#    std::clog << "\t <ef_search>: int,int,int,int...,int " << std::endl;
#    std::clog << "\t <k>: number of neighbors " << std::endl;
#    std::clog << "\t <Reorder ID>: 0 for no reordering, 1 for reordering"
#              << std::endl;
#    std::clog << "\t <Quantized>: 0 for no quantization, 1 for quantization"
#              << std::endl;


# Make sure we're at the top level directory.
cd "$(dirname "$0")/.."

# Make sure the user provided a dataset name.
if [ -z "$1" ]
    then
        echo "No dataset name provided. Please provide a dataset name."
        exit 1
fi

INDEX="data/$1/$1.index"
if [ ! -f "$INDEX" ]
    then
        echo "Index $INDEX does not exist. Please build the index first."
        exit 1
fi

# Now query the index
echo "Querying with dataset $1"
DATASET_NAME=$1
QUERY_DATASET_PATH="data/$DATASET_NAME/$DATASET_NAME.test.npy"
GROUNDTRUTH_DATASET_PATH="data/$DATASET_NAME/$DATASET_NAME.gtruth.npy"

# SPACE will be 0 if dataset name contains 'euclidean' otherwise 1
if [[ $DATASET_NAME == *"euclidean"* ]]
    then
        SPACE=0
    else
        SPACE=1
fi

./build/query_npy \
    $SPACE \
    $INDEX \
    $QUERY_DATASET_PATH \
    $GROUNDTRUTH_DATASET_PATH \
    100,200,300 \
    100 \
    0 \
    0