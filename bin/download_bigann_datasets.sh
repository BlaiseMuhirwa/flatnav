#!/bin/bash

PYTHON=$(which python3)

if [[ -z $PYTHON ]]; then
    echo "Python not found. Please install python3."
    exit 1
fi


# Create a list of ANNS benchmark datasets to download.
BIGANN_BENCHMARK_DATASETS=("bigann","deep","text2image", "msspacev")

# Default data directory
DATA_DIR="data"
CHUNK_SIZE=""
AXEL_CONNECTIONS="16"

function print_help() {
    echo "Usage: ./download_bigann_datasets.sh [options] --dataset <dataset>"
    echo ""
    echo "Options:"
    echo "  -d, --dir          Directory to store downloaded data (default: ./data)"
    echo "  --dataset          Dataset to download (required)"
    echo "  --chunk-size       Number of train vectors to keep (default: all)"
    echo "  -n, --connections  Number of parallel axel connections (default: 16)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Available datasets:"
    echo "  ${BIGANN_BENCHMARK_DATASETS[@]}"
    echo ""
    echo "Example Usage:"
    echo "  ./download_bigann_datasets.sh --dataset bigann"
    echo "  ./download_bigann_datasets.sh -d /mnt/data --dataset bigann"
    echo "  ./download_bigann_datasets.sh -d /mnt/data --dataset bigann --chunk-size 100000000"
    echo "  ./download_bigann_datasets.sh -d /mnt/data --dataset deep --chunk-size 10000000"
    exit 1
}



function download_dataset() {
    local dataset=$1

    # Build chunk-size argument for python scripts
    local chunk_arg=""
    if [ -n "${CHUNK_SIZE}" ]; then
        chunk_arg="--chunk-size ${CHUNK_SIZE}"
    fi

    # Skip download if directory data/dataset_name already exists.
    if [ -d "${DATA_DIR}/${dataset}" ]; then
        echo "${DATA_DIR}/${dataset} already exists. Skipping download."
        exit 0
    fi

    if [ -d "${DATA_DIR}/GT_10M" ]; then
        echo "${DATA_DIR}/GT_10M directory already exists. Skipping download."
    else
        mkdir -p "${DATA_DIR}"
        wget -O "${DATA_DIR}/GT_10M_v2.tgz" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M_v2.tgz && \
            tar -xzvf "${DATA_DIR}/GT_10M_v2.tgz" -C "${DATA_DIR}" && \
            rm -f "${DATA_DIR}/GT_10M_v2.tgz"
    fi

    if [ -d "${DATA_DIR}/GT_100M" ]; then
        echo "${DATA_DIR}/GT_100M directory already exists. Skipping download."
    else
        mkdir -p "${DATA_DIR}"
        wget -O "${DATA_DIR}/GT_100M_v2.tgz" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M_v2.tgz && \
            tar -xzvf "${DATA_DIR}/GT_100M_v2.tgz" -C "${DATA_DIR}" && \
            rm -f "${DATA_DIR}/GT_100M_v2.tgz"
    fi

    # Download 1B ground truth when fetching the full dataset (no chunk-size)
    if [ -z "${CHUNK_SIZE}" ]; then
        if [ -d "${DATA_DIR}/GT_1B" ]; then
            echo "${DATA_DIR}/GT_1B directory already exists. Skipping download."
        else
            mkdir -p "${DATA_DIR}"
            wget -O "${DATA_DIR}/GT_1B_v2.tgz" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_1B_v2.tgz && \
                tar -xzvf "${DATA_DIR}/GT_1B_v2.tgz" -C "${DATA_DIR}" && \
                rm -f "${DATA_DIR}/GT_1B_v2.tgz"
        fi
    fi

    mkdir -p "${DATA_DIR}/${dataset}"
    echo "Downloading ${dataset}..."

    if [ "${dataset}" == "bigann" ]; then
        axel -a -n "${AXEL_CONNECTIONS}" -o "${DATA_DIR}/${dataset}/bigann_base.1B.u8bin" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
        axel -a -n "${AXEL_CONNECTIONS}" -o "${DATA_DIR}/${dataset}/bigann_query.public.10K.u8bin" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin

        $PYTHON convert_bigann_datasets.py "${DATA_DIR}/${dataset}/bigann_query.public.10K.u8bin" queries
        $PYTHON convert_bigann_datasets.py "${DATA_DIR}/${dataset}/bigann_base.1B.u8bin" train ${chunk_arg}

    elif [ "${dataset}" == "deep" ]; then
        axel -a -n "${AXEL_CONNECTIONS}" -o "${DATA_DIR}/${dataset}/deep_base.1B.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
        axel -a -n "${AXEL_CONNECTIONS}" -o "${DATA_DIR}/${dataset}/deep_query.public.10K.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin

        $PYTHON convert_bigann_datasets.py "${DATA_DIR}/${dataset}/deep_query.public.10K.fbin" queries
        $PYTHON convert_bigann_datasets.py "${DATA_DIR}/${dataset}/deep_base.1B.fbin" train ${chunk_arg}

    elif [ "${dataset}" == "text2image" ]; then
        axel -a -n "${AXEL_CONNECTIONS}" -o "${DATA_DIR}/${dataset}/tti_base.1B.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
        axel -a -n "${AXEL_CONNECTIONS}" -o "${DATA_DIR}/${dataset}/tti_query.learn.50M.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin

        $PYTHON convert_bigann_datasets.py "${DATA_DIR}/${dataset}/tti_query.learn.50M.fbin" queries
        $PYTHON convert_bigann_datasets.py "${DATA_DIR}/${dataset}/tti_base.1B.fbin" train ${chunk_arg}

    elif [ "${dataset}" == "msspacev" ]; then
        set GIT_LFS_SKIP_SMUDGE=1
        git clone --recurse-submodules https://github.com/microsoft/SPTAG

        mv SPTAG/datasets/SPACEV1B/vectors.bin/ "${DATA_DIR}/${dataset}/"
        mv SPTAG/datasets/SPACEV1B/query.bin "${DATA_DIR}/${dataset}/"

        $PYTHON convert_spacev_dataset.py "${DATA_DIR}/${dataset}/query.bin" queries
        $PYTHON convert_spacev_dataset.py "${DATA_DIR}/${dataset}/vectors.bin" train ${chunk_arg}

    else
        echo "Invalid Dataset Choice!"

    fi

    $PYTHON convert_bigann_datasets.py "${DATA_DIR}/GT_10M/${dataset}-10M" gt --output-dir "${DATA_DIR}/${dataset}"
    $PYTHON convert_bigann_datasets.py "${DATA_DIR}/GT_100M/${dataset}-100M" gt --output-dir "${DATA_DIR}/${dataset}"

    # Convert 1B ground truth if available
    if [ -z "${CHUNK_SIZE}" ] && [ -d "${DATA_DIR}/GT_1B" ]; then
        $PYTHON convert_bigann_datasets.py "${DATA_DIR}/GT_1B/${dataset}-1B" gt --output-dir "${DATA_DIR}/${dataset}"
    fi

}

# Parse arguments
DATASET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_help
            ;;
        -d|--dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -n|--connections)
            AXEL_CONNECTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    print_help
fi

download_dataset "$DATASET"
exit 0
