#!/bin/bash

PYTHON=$(which python3)

if [[ -z $PYTHON ]]; then
    echo "Python not found. Please install python3."
    exit 1
fi

# Resolve this script's directory and the repo root so the Python converters can
# be invoked regardless of the current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
CONVERT_BIGANN="${REPO_ROOT}/convert_bigann_datasets.py"

CONVERT_SPACEV="${REPO_ROOT}/convert_spacev_dataset.py"

# All downloaded data, intermediate clones, and converted .npy outputs land
# under DATA_DIR. Override via environment variable configuration. 
DATA_DIR="${DATA_DIR:-./data}"
export DATA_DIR
mkdir -p "${DATA_DIR}"


# Create a list of ANNS benchmark datasets to download.
BIGANN_BENCHMARK_DATASETS=("bigann","deep","text2image", "msspacev")

function print_help() {
    echo "Usage: ./download_bigann_datasets.sh --dataset-name <name> [--axel-connections <n>]"
    echo ""
    echo "Available datasets:"
    echo "${BIGANN_BENCHMARK_DATASETS[@]}"
    echo ""
    echo "Arguments:"
    echo "  --dataset-name <name>       One of the datasets listed above (required)."
    echo "  --axel-connections <n>       Number of parallel axel connections (default: 4)."
    echo "                              Ignored for msspacev (uses git-lfs, not axel)."
    echo ""
    echo "Output directory (override with DATA_DIR env var):"
    echo "  ${DATA_DIR}"
    echo ""
    echo "Example Usage:"
    echo "  ./download_bigann_datasets.sh --dataset-name bigann"
    echo "  ./download_bigann_datasets.sh --dataset-name deep --axel-connections 8"
    echo "  DATA_DIR=/scratch/me/data ./download_bigann_datasets.sh --dataset-name msspacev"
    exit 1
}



function download_dataset() {
    local dataset=$1
    # Number of parallel axel connections; reasonable default of 4.
    local connections="${2:-4}"

    # Validate that connections is a positive integer.
    if ! [[ "${connections}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid connections value: '${connections}'. Must be a positive integer."
        exit 1
    fi

    # axel is required to fetch the bigann/deep/text2image datasets, but not
    # msspacev (which is pulled via git-lfs). Only require it when needed.
    if [ "${dataset}" != "msspacev" ] && ! command -v axel >/dev/null 2>&1; then
        echo "axel not found. Please install axel first."
        exit 1
    fi

    # Skip download if the dataset directory already exists.
    if [ -d "${DATA_DIR}/${dataset}" ]; then
        echo "${DATA_DIR}/${dataset} already exists. Skipping download."
        exit 0
    fi

    if [ -d "${DATA_DIR}/GT_10M" ]; then
       echo "GT_10M directory already exists. Skipping download."
    else
       wget -O "${DATA_DIR}/GT_10M_v2.tgz" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_10M_v2.tgz \
           && tar -xzvf "${DATA_DIR}/GT_10M_v2.tgz" -C "${DATA_DIR}"
    fi

    if [ -d "${DATA_DIR}/GT_100M" ]; then
       echo "GT_100M directory already exists. Skipping download."
    else
       wget -O "${DATA_DIR}/GT_100M_v2.tgz" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M_v2.tgz \
           && tar -xzvf "${DATA_DIR}/GT_100M_v2.tgz" -C "${DATA_DIR}"
    fi

    mkdir -p "${DATA_DIR}/${dataset}"
    echo "Downloading ${dataset}..."

    if [ ${dataset} == "bigann" ]; then
        axel -a -n "${connections}" -o "${DATA_DIR}/${dataset}/bigann_base.1B.u8bin" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin
        axel -a -n "${connections}" -o "${DATA_DIR}/${dataset}/bigann_query.public.10K.u8bin" https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin

        $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/${dataset}/bigann_query.public.10K.u8bin" queries
        $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/${dataset}/bigann_base.1B.u8bin" train

    elif [ ${dataset} == "deep" ]; then
        axel -a -n "${connections}" -o "${DATA_DIR}/${dataset}/deep_base.1B.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin
        axel -a -n "${connections}" -o "${DATA_DIR}/${dataset}/deep_query.public.10K.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin

        $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/${dataset}/deep_query.public.10K.fbin" queries
        $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/${dataset}/deep_base.1B.fbin" train

    elif [ ${dataset} == "text2image" ]; then
        axel -a -n "${connections}" -o "${DATA_DIR}/${dataset}/tti_base.1B.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin
        axel -a -n "${connections}" -o "${DATA_DIR}/${dataset}/tti_query.learn.50M.fbin" https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.learn.50M.fbin

        $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/${dataset}/tti_query.learn.50M.fbin" queries
        $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/${dataset}/tti_base.1B.fbin" train

    elif [ ${dataset} == "msspacev" ]; then
        if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
            echo "git-lfs not found. Please install git-lfs first."
            exit 1
        fi
        # SPACEV1B data is stored via git-lfs in the main SPTAG repo (NOT in a
        # submodule), so skip --recurse-submodules. Clone without smudging LFS
        # (GIT_LFS_SKIP_SMUDGE=1), then selectively pull only the paths we need.
        if [ ! -d "${DATA_DIR}/SPTAG" ]; then
            GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/microsoft/SPTAG "${DATA_DIR}/SPTAG"
        fi
        ( cd "${DATA_DIR}/SPTAG" && \
          git lfs pull --include="datasets/SPACEV1B/vectors.bin/*,datasets/SPACEV1B/query.bin" )

        mv "${DATA_DIR}/SPTAG/datasets/SPACEV1B/vectors.bin" "${DATA_DIR}/${dataset}/"
        mv "${DATA_DIR}/SPTAG/datasets/SPACEV1B/query.bin" "${DATA_DIR}/${dataset}/"

        $PYTHON "${CONVERT_SPACEV}" "${DATA_DIR}/${dataset}/query.bin" queries
        $PYTHON "${CONVERT_SPACEV}" "${DATA_DIR}/${dataset}/vectors.bin" train

    else
        echo "Invalid Dataset Choice!"

    fi

    $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/GT_10M/${dataset}-10M" gt
    $PYTHON "${CONVERT_BIGANN}" "${DATA_DIR}/GT_100M/${dataset}-100M" gt

}

# Parse named arguments.
DATASET_NAME=""
AXEL_CONNECTIONS="4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --axel-connections)
            AXEL_CONNECTIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            print_help
            ;;
    esac
done

if [[ -z "${DATASET_NAME}" ]]; then
    echo "Error: --dataset-name is required."
    print_help
fi

download_dataset "${DATASET_NAME}" "${AXEL_CONNECTIONS}"
exit 0
