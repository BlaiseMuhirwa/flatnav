#!/bin/bash 


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
    echo "Usage: ./download_anns_datasets.sh <dataset> [--normalize]"
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
    local normalize=$2

    # Skip download if directory data/dataset_name already exists.
    if [ -d "data/${dataset}" ]; then
        echo "data/${dataset} already exists. Skipping download."
        exit 0
    fi

    if [ -f "${dataset}.hdf5" ]; then
        echo "${dataset}.hdf5 already exists. Skipping download."
    else
        echo "Downloading ${dataset}..."
        curl -L -o ${dataset}.hdf5 http://ann-benchmarks.com/${dataset}.hdf5
    fi

    # Create directory and move dataset to data/dataset_name.
    mkdir -p data/${dataset}
    mv ${dataset}.hdf5 data/${dataset}/${dataset}.hdf5

    # Create a set of training, query and groundtruth files by running the python 
    # script dump.py on the downloaded dataset. If normalize is set to 1, then pass 
    # the --normalize flag to dump.py.

    if [ ${normalize} -eq 1 ]; then
        python dump.py data/${dataset}/${dataset}.hdf5 --normalize
    else
        python dump.py data/${dataset}/${dataset}.hdf5
    fi
}


# If the first argument is -h or --help, then print help and exit.
if [[ $1 == "-h" || $1 == "--help" ]]; then
    print_help
fi



# Check if a user ran the script like this: ./download_anns_datasets.sh <dataset> --normalize 
# If so, then download only the specified dataset and normalize it.
# If they just ran the script like this: ./download_anns_datasets.sh <dataset>, then 
# download only the specified dataset and do not normalize it.
if [[ $# -eq 2 ]]; then
    if [[ $2 == "--normalize" ]]; then
        download_dataset $1 1
        exit 0
    fi
elif [[ $# -eq 1 ]]; then 
    download_dataset $1 0 
    exit 0 
fi


echo "No dataset specified. Downloading all datasets."

# Download each dataset in the list.
for dataset in "${ANN_BENCHMARK_DATASETS[@]}"; do
    # If dataset name contains a substring "angular", then normalize the dataset.
    if [[ $dataset == *"angular"* ]]; then
        download_dataset ${dataset} 1
        continue
    fi
    download_dataset ${dataset} 0
done