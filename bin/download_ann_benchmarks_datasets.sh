#!/bin/bash 

PYTHON=$(which python3)

if [[ -z $PYTHON ]]; then 
    echo "Python not found. Please install python3."
    exit 1
fi

# Make sure we are in this directory before runnin


# Create a list of ANNS benchmark datasets to download.
ANN_BENCHMARK_DATASETS=("mnist-784-euclidean" 
                        "sift-128-euclidean" 
                        "glove-25-angular" 
                        "glove-100-angular" 
                        "glove-50-angular" 
                        "glove-200-angular" 
                        "deep-image-96-angular" 
                        "gist-960-euclidean" 
                        "nytimes-256-angular")

function print_help() {
    echo "Usage: ./download_ann_benchmarks_datasets.sh --dataset-name <name> [--normalize]"
    echo ""
    echo "Available datasets:"
    echo "${ANN_BENCHMARK_DATASETS[@]}"
    echo ""
    echo "Arguments:"
    echo "  --dataset-name <name>   One of the datasets listed above (required)."
    echo "  --normalize             Normalize the dataset vectors."
    echo ""
    echo "Example Usage:"
    echo "  ./download_ann_benchmarks_datasets.sh --dataset-name mnist-784-euclidean"
    echo "  ./download_ann_benchmarks_datasets.sh --dataset-name glove-25-angular --normalize"
    exit 1
}


function download_dataset() {
    # Downloads a single benchmark dataset for Approximate Nearest Neighbor
    # Search (ANNS). Datasets are downloaded from http://ann-benchmarks.com/
    # and are stored in the data/ directory.

    # Usage: ./download_dataset.sh <dataset_name> <normalize>

    local dataset=$1
    local normalize=$2

    # Skip download if directory data/dataset_name already exists.
    if [ -d "data/${dataset}" ]; then
        echo "data/${dataset} already exists. Skipping download."
        exit 0
    fi


    echo "Downloading ${dataset}..."
    curl -L -o ${dataset}.hdf5 http://ann-benchmarks.com/${dataset}.hdf5

    # Create directory and move dataset to data/dataset_name.
    mkdir -p data/${dataset}
    mv ${dataset}.hdf5 data/${dataset}/${dataset}.hdf5

    # Create a set of training, query and groundtruth files by running the python 
    # script convert_ann_benchmark_datasets.py on the downloaded dataset. If normalize is set to 1, then pass 
    # the --normalize flag to convert_ann_benchmark_datasets.py.

    if [ ${normalize} -eq 1 ]; then
        $PYTHON convert_ann_benchmark_datasets.py data/${dataset}/${dataset}.hdf5 --normalize
    else
        $PYTHON convert_ann_benchmark_datasets.py data/${dataset}/${dataset}.hdf5
    fi
}


# Parse named arguments.
DATASET_NAME=""
NORMALIZE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            print_help
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --normalize)
            NORMALIZE=1
            shift
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

download_dataset "${DATASET_NAME}" "${NORMALIZE}"
exit 0
