# Reproduction of FlatNav vs. HNSWLib Benchmarking Experiments

## Overview
This document provides instructions for reproducing the experimental results comparing our non-hierarchical NSW implementation in FlatNav to the popular open source [HNSWLib](https://github.com/nmslib/hnswlib) library which utilizes a layered hierarchical graph. 

To enable relatively seamless reproducibility, we require the users to do the following:

* A machine with [docker](https://www.docker.com/) installed (and sufficient RAM to build and query indexes for a given workload). If you prefer to run the experiments without docker, we will add additional instructions shortly (though we highly recommend using the docker approach for complete consistency with our reported results)

* Download and preprocess the benchmark datasets into a certain folder named as `data` in the top-level directory of the `flatnav` repository. We provide more detailed instructions for this step in the next sections. 

* Executing the command `./bin/docker-run.sh <make-target>` from top-level directory of the flatnav repository. The `<make-target>` argument specifies the parameters of the benchmarking job to execute. We specify the make targets in the file [Makefile](/experiments/Makefile).

## Example Commands

Assuming you have docker installed and have prepared a benchmark dataset into the `data` directory, one can reproduce any one of our experiments specified in `experiments/Makefile`. For example, the following command will benchmark `flatnav` on the `gist` dataset. 

```shell
./bin/docker-run.sh gist-bench-flatnav
```

The analogous `hnswlib` benchmarking job on `gist` can be executed with a similar command. Again, the details of this make target are specified in the [Makefile](/experiments/Makefile). 

```shell
./bin/docker-run.sh gist-bench-hnsw
```

**NOTE:** We currently mount the data as a volume so that the experiment runner script has access 
to the dataset. What this means for you is that you have to place the dataset you want to use under the 
[data](/data/) subdirectory. Then, when you define a new target, specify the data path as `/root/data/<dataset-name>`. For instance, for the `sift-bench` we specify the dataset path like this:

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

## Input Data Format

Our experimental benchmark scripts require three input data arguments in numpy (`.npy`) format. 

* A `--train` file representing the vectors of each item in the data collection used to build the search index. This is expected to be a numpy array of dimension $N \times d$ where $N$ is the database size and $d$ is the vector dimension

* A `--queries` file representing the query vectors to use to search against the index. This file is expected to be a numpy array of dimension $Q \times d$ where $Q$ is the number of queries. 

* A `--gtruth` file consisting of the true $k$ nearest neighbors for each corresponding query vector. This file is expected to be an integer numpy array of dimension $Q \times k$ where $k$ is the number of near neighbors to return (we default to 100 in our experiments). Each element of this array is expected to be an integer in the range $[0, N-1]$ representing items in the index. 

## Preparing Datasets from ANN-Benchmarks

[ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) provide HDF5 files for a standard benchmark of near-neighbor datasets, queries and ground-truth results. Our experiment runner expects `.npy` files instead of HDF5 so we provide a helper script to download ANN-Benchmarks and prepare the necessary numpy files.

To generate an [ANNS benchmark datasets](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file#data-sets), run the following script

```shell
$ ./bin/download_anns_datasets.sh <dataset-name> [--normalize]
```

__IMPORTANT:__ For datasets that use the angular/cosine similarity, you will need to use `--normalize` option so that the distances are computed correctly. 

Available dataset names include:

```shell
_ mnist-784-euclidean
_ sift-128-euclidean
_ glove-25-angular
_ glove-50-angular
_ glove-100-angular
_ glove-200-angular
_ deep-image-96-angular
_ gist-960-euclidean
_ nytimes-256-angular
```

## Preparing Datasets from Big-ANN Benchmarks

[Big-ANN Benchmarks](https://big-ann-benchmarks.com/neurips21.html) Is a more recent set of ANN benchmark datasets focused on extremely large scales. Specifically, Big-ANN Benchmarks provides access to embedding datasets and ground truth near neighbor sets for 10M, 100M, and 1B vectors. In our benchmarks, we focus on the 10M and 100M datasets due to computational resource constraints. To reproduce our results in the paper, we provide a helper script to download the 10M and 100M Big-ANN Benchmark datasets and convert them into the numpy format that `flatnav` expects. 

To process a Big-ANN benchmark datasets, please run the following script. This script will download and process the 10M and 100M versions of the given dataset, including the ground truth file. Then, one can run a benchmarking job as described above (these Big-ANN dataset configurations are already specified in the Makefile)

```shell
$ ./bin/download_bigann_datasets.sh <dataset-name>
```

The available dataset names include:

```shell
- bigann
- deep
- text2image
```
These available datasets include three of the four Big-ANN Benchmark datasets we evaluate in our paper. The Microsoft SpaceV dataset is no longer available through the same public download link, so we do not include it here. We will provide instructions on how to access the SpaceV datasets shortly. 
