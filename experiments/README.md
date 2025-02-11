# Reproduction of FlatNav vs. HNSWLib Benchmarking Experiments

## Overview
This document provides instructions for reproducing the experimental results comparing our non-hierarchical NSW implementation in FlatNav to the popular open source [HNSWLib](https://github.com/nmslib/hnswlib) library which utilizes a layered hierarchical graph. 

As a first step, please clone the `flatnav` GitHub repository which contains all of the relevant experimental scripts

```shell
git clone https://github.com/BlaiseMuhirwa/flatnav.git --recurse-submodules
```

To enable relatively seamless reproducibility, we require users to do the following:

* A machine with [docker](https://www.docker.com/) installed (and sufficient RAM to build and query indexes for a given workload). In our experiments, we use an AWS `c6i.8xlarge` instance for all benchmarks with less than 100M points. This machine comes up with 64 GiB of RAM and an Intel Xeon 8375C (Ice Lake) processor. For our 100M-scale experiments, we use a machine with 1TB of RAM and an AMD EPYC 9J14 96-Core Processor. 

* Download and preprocess the benchmark datasets into a certain folder named as `data` in the top-level directory of the `flatnav` repository. We provide more detailed instructions for this step in the next sections. 

* Executing the command `./bin/docker-run.sh <make-target>` from top-level directory of the flatnav repository. The `<make-target>` argument specifies the parameters of the benchmarking job to execute. We specify the make targets in the file [Makefile](/experiments/Makefile).

## Example Commands

Assuming you have docker installed, one can download any of the benchmark datases and reproduce any one of our experiments specified in `experiments/Makefile`. For example, the following command wills download the gist dataset and then benchmark `flatnav` and then `hnswlib` on the `gist` dataset. 

```shell
./bin/download_ann_benchmarks_datasets.sh gist-960-euclidean

./bin/docker-run.sh gist-bench-flatnav

./bin/docker-run.sh gist-bench-hnsw
```

The details of all the benchmark make targets are specified in the [Makefile](/experiments/Makefile). 

**NOTE:** We currently mount the data as a volume so that the experiment runner script has access 
to the dataset. What this means for you is that you have to place the dataset you want to use under the 
[data](/data/) subdirectory. Our data downloading scripts (described below) do this automatically. When you define a new target, specify the data path as `/root/data/<dataset-name>`. For instance, for the `sift-bench` we specify the dataset path like this:

```
sift-bench: 
	poetry run python run-benchmark.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename sift.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 \
		--ef-search 100 200 300 \
		--metric l2 
```

You may also want to save the experiment logs to a file on disk. You can do so by running 
```
> ./bin/docker-test.sh sift-bench > logs.txt 2>& 1
```
### Viewing Output Metrics

Once you have run a benchmarking job to completion, the experiment runner will save a set of plots under the `metrics` directory in the top level of the `flatnav` repo. These plots include, amongst others, the latency vs. recall tradeoff curves that we report in the paper. We also save the raw data used to generate these plots in the file `metrics/metrics.json`. 

## Full List of Commands

If you would like to understand more of the details of our data preparation scripts, please see the remaining sections below. If you would simply like to reproduce our benchmarking results for all 13 ANN Benchmarks and Big ANN Benchmark datasets considered in the paper, we list the specific commands per datasets below. 

### MNIST Dataset
```shell
./bin/download_ann_benchmarks_datasets.sh mnist-784-euclidean

./bin/docker-run.sh mnist-bench-flatnav

./bin/docker-run.sh mnist-bench-hnsw
```

### GIST Dataset
```shell
./bin/download_ann_benchmarks_datasets.sh gist-960-euclidean

./bin/docker-run.sh gist-bench-flatnav

./bin/docker-run.sh gist-bench-hnsw
```

### Yandex Deep Dataset (ANN Benchmarks)
```shell
./bin/download_ann_benchmarks_datasets.sh deep-image-96-angular --normalize

./bin/docker-run.sh deep-image-bench-flatnav

./bin/docker-run.sh deep-image-bench-hnsw
```

### SIFT Dataset

```shell
./bin/download_ann_benchmarks_datasets.sh sift-128-euclidean

./bin/docker-run.sh sift-bench-flatnav

./bin/docker-run.sh sift-bench-hnsw
```

### NYTimes Dataset

```shell
./bin/download_ann_benchmarks_datasets.sh nytimes-256-angular --normalize

./bin/docker-run.sh nytimes-bench-flatnav

./bin/docker-run.sh nytimes-bench-hnsw
```

### Glove 25 Dataset

```shell
./bin/download_ann_benchmarks_datasets.sh glove-25-angular --normalize

./bin/docker-run.sh glove25-bench-flatnav

./bin/docker-run.sh glove25-bench-hnsw
```

### Glove 50 Dataset

```shell
./bin/download_ann_benchmarks_datasets.sh glove-50-angular --normalize

./bin/docker-run.sh glove50-bench-flatnav

./bin/docker-run.sh glove50-bench-hnsw
```

### Glove 100 Dataset

```shell
./bin/download_ann_benchmarks_datasets.sh glove-100-angular --normalize

./bin/docker-run.sh glove100-bench-flatnav

./bin/docker-run.sh glove100-bench-hnsw
```

### Glove 200 Dataset

```shell
./bin/download_ann_benchmarks_datasets.sh glove-200-angular --normalize

./bin/docker-run.sh glove200-bench-flatnav

./bin/docker-run.sh glove200-bench-hnsw
```

### Bigann 10M and 100M Datasets
```shell
./bin/download_bigann_datasets.sh bigann

./bin/docker-run.sh bigann-10m-bench-flatnav

./bin/docker-run.sh bigann-10m-bench-hnsw

./bin/docker-run.sh bigann-100m-bench-flatnav

./bin/docker-run.sh bigann-100m-bench-hnsw
```

### Yandex Deep 10M and 100M Datasets
```shell
./bin/download_bigann_datasets.sh deep

./bin/docker-run.sh yandex-deep-10m-bench-flatnav

./bin/docker-run.sh yandex-deep-10m-bench-hnsw

./bin/docker-run.sh yandex-deep-100m-bench-flatnav

./bin/docker-run.sh yandex-deep-100m-bench-hnsw
```

### Yandex Text-to-Image 10M and 100M Datasets
```shell
./bin/download_bigann_datasets.sh text2image

./bin/docker-run.sh tti-10m-bench-flatnav

./bin/docker-run.sh tti-10m-bench-hnsw

./bin/docker-run.sh tti-100m-bench-flatnav

./bin/docker-run.sh tti-100m-bench-hnsw
```

### Microsoft SpaceV 10M and 100M Datasets
```shell
./bin/download_bigann_datasets.sh msspacev

./bin/docker-run.sh spacev-10m-bench-flatnav

./bin/docker-run.sh spacev-10m-bench-hnsw

./bin/docker-run.sh spacev-100m-bench-flatnav

./bin/docker-run.sh spacev-100m-bench-hnsw
```

## Input Data Format

Our experimental benchmark scripts require three input data arguments in numpy (`.npy`) format. 

* A `--train` file representing the vectors of each item in the data collection used to build the search index. This is expected to be a numpy array of dimension $N \times d$ where $N$ is the database size and $d$ is the vector dimension

* A `--queries` file representing the query vectors to use to search against the index. This file is expected to be a numpy array of dimension $Q \times d$ where $Q$ is the number of queries. 

* A `--gtruth` file consisting of the true $k$ nearest neighbors for each corresponding query vector. This file is expected to be an integer numpy array of dimension $Q \times k$ where $k$ is the number of near neighbors to return (we default to 100 in our experiments). Each element of this array is expected to be an integer in the range $[0, N-1]$ representing items in the index. 

## Preparing Datasets from ANN-Benchmarks

[ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) provide HDF5 files for a standard benchmark of near-neighbor datasets, queries and ground-truth results. Our experiment runner expects `.npy` files instead of HDF5 so we provide a helper script to download ANN-Benchmarks and prepare the necessary numpy files.

To generate an [ANNS benchmark datasets](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file#data-sets), run the following script

```shell
./bin/download_ann_benchmarks_datasets.sh <dataset-name> [--normalize]
```

__IMPORTANT:__ For datasets that use the angular/cosine similarity, you will need to use `--normalize` option so that the distances are computed correctly. 

Available dataset names include:

```shell
- mnist-784-euclidean
- sift-128-euclidean
- glove-25-angular
- glove-50-angular
- glove-100-angular
- glove-200-angular
- deep-image-96-angular
- gist-960-euclidean
- nytimes-256-angular
```

## Preparing Datasets from Big-ANN Benchmarks

[Big-ANN Benchmarks](https://big-ann-benchmarks.com/neurips21.html) Is a more recent set of ANN benchmark datasets focused on extremely large scales. Specifically, Big-ANN Benchmarks provides access to embedding datasets and ground truth near neighbor sets for 10M, 100M, and 1B vectors. In our benchmarks, we focus on the 10M and 100M datasets due to computational resource constraints. To reproduce our results in the paper, we provide a helper script to download the 10M and 100M Big-ANN Benchmark datasets and convert them into the numpy format that `flatnav` expects. 

To process a Big-ANN benchmark datasets, please run the following script. This script will download and process the 10M and 100M versions of the given dataset, including the ground truth file. Then, one can run a benchmarking job as described above (these Big-ANN dataset configurations are already specified in the Makefile)

```shell
./bin/download_bigann_datasets.sh <dataset-name>
```

The available dataset names include:

```shell
- bigann
- deep
- text2image
- msspacev
```
Note that the Microsoft SpaceV Dataset is no longer available via the public link on the [Big-ANN Benchmarks](https://big-ann-benchmarks.com/neurips21.html) website. We instead access this dataset through the [SPTAG](https://github.com/microsoft/SPTAG) GitHub repository.
