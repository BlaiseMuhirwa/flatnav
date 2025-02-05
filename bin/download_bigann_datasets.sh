#!/bin/bash 

PYTHON=$(which python3)

if [[ -z $PYTHON ]]; then 
    echo "Python not found. Please install python3."
    exit 1
fi


# Create a list of ANNS benchmark datasets to download.
BIGANN_BENCHMARK_DATASETS=("bigann","deep","text2image", "msspacev")

function print_help() {
    echo "Usage: ./download_bigann_datasets.sh <dataset>"
    echo ""
    echo "Available datasets:"
    echo "${BIGANN_BENCHMARK_DATASETS[@]}"
    echo ""
    echo "Example Usage:"
    echo "  ./download_bigann_datasets.sh bigann"
    echo "  ./download_bigann_datasets.sh deep"
    exit 1
}



function download_dataset() {
    local dataset=$1

    # Skip download if directory data/dataset_name already exists.
    if [ -d "data/${dataset}" ]; then
        echo "data/${dataset} already exists. Skipping download."
        exit 0
    fi

    if [ -d "GT_10M" ]; then
       echo "GT_10M directory already exists. Skipping download."
    else
       wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M_v2.tgz && tar -xzvf GT_10M_v2.tgz
    fi

    if [ -d "GT_100M" ]; then
       echo "GT_100M directory already exists. Skipping download."
    else
       wget https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M_v2.tgz && tar -xzvf GT_100M_v2.tgz
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
    
    elif [ ${dataset} == "deep" ]; then
	axel -a -o deep_base.1B.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
        axel -a -o deep_query.public.10K.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin

        mv deep_base.1B.fbin data/${dataset}/deep_base.1B.fbin
	mv deep_query.public.10K.fbin data/${dataset}/deep_query.public.10K.fbin

        $PYTHON convert_bigann_datasets.py data/${dataset}/deep_query.public.10K.fbin queries
	$PYTHON convert_bigann_datasets.py data/${dataset}/deep_base.1B.fbin train
        

    elif [ ${dataset} == "text2image" ]; then
	axel -a -o tti_base.1B.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
	axel -a -o tti_query.learn.50M.fbin https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin
        
        mv tti_base.1B.fbin data/${dataset}/tti_base.1B.fbin
        mv tti_query.learn.50M.fbin data/${dataset}/tti_query.learn.50M.fbin

        $PYTHON convert_bigann_datasets.py data/${dataset}/tti_query.learn.50M.fbin queries
        $PYTHON convert_bigann_datasets.py data/${dataset}/tti_base.1B.fbin train
    
    elif [ ${dataset} == "msspacev" ]; then
       set GIT_LFS_SKIP_SMUDGE=1
       git clone --recurse-submodules https://github.com/microsoft/SPTAG
       
       mv SPTAG/datasets/SPACEV1B/vectors.bin/ data/${dataset}/
       mv SPTAG/datasets/SPACEV1B/query.bin data/${dataset}/

       $PYTHON convert_spacev_dataset.py data/${dataset}/query.bin queries
       $PYTHON convert_spacev_dataset.py data/${dataset}/vectors.bin train 

    else
	echo "Invalid Dataset Choice!"

    fi

    $PYTHON convert_bigann_datasets.py GT_10M/${dataset}-10M gt
    $PYTHON convert_bigann_datasets.py GT_100M/${dataset}-100M gt

}

# If the first argument is -h or --help, then print help and exit.
if [[ $1 == "-h" || $1 == "--help" ]]; then
    print_help
fi


download_dataset $1
exit 0
