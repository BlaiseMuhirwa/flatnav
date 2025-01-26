#!/bin/bash 

PYTHON=$(which python3)

if [[ -z $PYTHON ]]; then 
    echo "Python not found. Please install python3."
    exit 1
fi

# Make sure we are in this directory before runnin


# Create a list of ANNS benchmark datasets to download.
ANN_BENCHMARK_DATASETS=("bigann")

function print_help() {
    echo "Usage: ./download_big_ann_datasets.sh <dataset> [--normalize]"
    echo ""
    echo "Available datasets:"
    echo "${ANN_BENCHMARK_DATASETS[@]}"
    echo ""
    echo "Example Usage:"
    echo "  ./download_anns_datasets.sh mnist-784-euclidean"
    echo "  ./download_anns_datasets.sh glove-25-angular --normalize"
    exit 1
}


function download_dataset() {
    # Downloads a single benchmark dataset for Approximate Nearest Neighbor
    # Search (ANNS). Datasets are downloaded from http://ann-benchmarks.com/
    # and are stored in the data/ directory.

    # Usage: ./download_dataset.sh <dataset_name> <normalize>

    local dataset=$1
    local size=$2

    # Skip download if directory data/dataset_name already exists.
    if [ -d "data/${dataset}" ]; then
        echo "data/${dataset} already exists. Skipping download."
        exit 0
    fi

    mkdir -p data/${dataset}
    echo "Downloading ${dataset}..."
    
    if [ ${dataset} == "bigann" ]; then
        axel -a -o bigann_base.1B.u8bin https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
        axel -a -o bigann_query.public.10K.u8bin https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin
   	mv bigann_base.1B.u8bin data/${dataset}/bigann_base.1B.u8bin
        mv bigann_query.public.10K.u8bin data/${dataset}/bigann_query.public.10K.u8bin	
    
	$PYTHON convert_bigann_datasets.py data/${dataset}/bigann_query.public.10K.u8bin queries
	$PYTHON convert_bigann_datasets.py data/${dataset}/bigann_base.1B.u8bin train

    
    elif [ ${dataset} == "yandex-deep" ]; then
	axel -a -o deep_base.1B.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
        axel -a -o deep_query.public.10K.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin

        mv deep_base.1B.fbin data/${dataset}/deep_base.1B.fbin
	mv deep_query.public.10K.fbin data/${dataset}/deep_query.public.10K.fbin

        $PYTHON convert_bigann_datasets.py data/${dataset}/deep_query.public.10K.fbin queries
	$PYTHON convert_bigann_datasets.py data/${dataset}/deep_base.1B.fbin train


    elif [ ${dataset} == "yandex-tti" ]; then
	axel -a -o tti_base.1B.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
	axel -a -o tti_query.learn.50M.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
        
        mv tti_base.1B.fbin data/${dataset}/tti_base.1B.fbin
        mv tti_query.learn.50M.fbin data/${dataset}/tti_query.learn.50M.fbin

        $PYTHON convert_bigann_datasets.py data/${dataset}/tti_query.learn.50M.fbin queries
        $PYTHON convert_bigann_datasets.py data/${dataset}/tti_base.1B.fbin train 
    else
	echo "Invalid Choice!"

    fi

#    if [ ${size} == "10M" ]; then
#    	wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M_v2.tgz && tar -xzvf GT_10M_v2.tgz
#
#    elif [ ${size} == "100M" ]; then
#	wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M_v2.tgz && tar -xzvf GT_100M_v2.tgz
#    else
#	echo "Invalid choice!"
#    fi
#
    
}


# If the first argument is -h or --help, then print help and exit.
if [[ $1 == "-h" || $1 == "--help" ]]; then
    print_help
fi


download_dataset $1 $2 
exit 0
