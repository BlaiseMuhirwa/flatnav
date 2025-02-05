
# Reproducibility of the Hub-Highway Experiments

## Overview

This document provides instructions for reproducing the experimental results related to the **hub-highway hypothesis** as detailed in our [paper](https://arxiv.org/pdf/2412.01940).

The reproduction is broken into 4 main steps, namely 

* Generating experimental datasets
* Running experiments to show the skewness of the node access distribution 
* Running experiments to show that hub nodes form a connected subgraph
* Running experiments to show that hub nodes allow queries to traverse the graph faster 

To enable relatively seamless reproducibility, we require the users to 

* Have a machine with [docker](https://www.docker.com/) installed (and sufficient RAM to build and query indexes for a given workload). 


**NOTE**: All results concerning the hubness experiments, are on a branch that can be found [here](https://github.com/BlaiseMuhirwa/flatnav/tree/hub-highway-experiments).

### Generating Experimental Datasets

In our experiments, we use both [ANN benchmark datasets](https://github.com/erikbern/ann-benchmarks) and synthetic datsets. The latter consists of uniform IID Gaussian vectors with increasing dimensionality (1, 2, 4, 8, 16, 32, 64, 128, 1024 and 1536) for both Euclidean and Cosine similarity.

To generate the datasets, execute the [generated-datasets](https://github.com/BlaiseMuhirwa/flatnav/blob/02113b5cf55ff96a6fa993fbb600d5f807d7d6b8/experiments/Makefile#L449) target inside the docker container by running

```shell
./bin/docker-run.sh generate-datasets
```
This will generate the datasets of size 1M and 10k queries and it will serialize them to disk. For instance, in the Euclidean case, for d=128, the following three Numpy files will be generated

* normal-128-euclidean.train.npy
* normal-128-euclidean.test.npy
* normal-128-euclidean.gtruth.npy

By default the directory structure for the data will be `./data/normal-128-euclidean/normal-128-euclidean.train.npy`. You can optionally set an environment variable that instructs docker about where to put your data prior to running the container with 

```shell
export DATA_DIR=<custom-data-directory>
./bin/docker-run.sh generate-datasets
```

### Running Experiments to show the Skewness of the Node Access Distribution 

The first set of experiments we run show that the distribution of the number of times each node in a FlatNav index is skewed to the right. We provide a way to generate this node access distribution for each dataset. Run

```shell
./bin/docker-run.sh generate-distributions
```

to generate this distribution for each one of the datasets generated above, and including some ANN benchmark datasets. 
This will, in addition, generate the outdegree table (graph representation) for each one of the datasets. 

The default directory structure for the data will be `./node-access-distribution/normal-128-euclidean_node_access_counts.json` and 
`./node-access-distribution/normal-128-euclidean_outdegree_table.pkl`. 

The `node-access-distribution` directory is created automatically and mounted as a volume. 

With this distribution saved, we can reproduce the plots in Figure 9 of the paper by running 

```shell
./bin/docker-run.sh kde-plots
```

### Running Experiments to Show that Hub Nodes are More Connected than Random Nodes in the Indexes

To show the connectivity of hub nodes, we use a one-sided Mann-Whitney U test and a two-sample t-test. 

The setup for the hypothesis tests is laid down [here](https://github.com/BlaiseMuhirwa/flatnav/blob/dca1ebf7be4caa991b1cce38f5b983d8bb0c2aab/experiments/statistical_test.py#L42-L69)

To run these tests, execute 

```shell
./bin/docker-run.sh run-hypothesis-test
```

This will save the p-values (and other metrics, such as effect size) under `./metrics/hypothesis_tests.json`. 

### Running Experiments to Show that Hub Nodes allow Queries to Traverse the Graph Faster 

In this set of experiments (Figure 10) in the paper, we show that queries tend to spend the earlier part of the search navigating through hub nodes, which propels them into disparate regions of the graph. 

For these experiments, we track the sequence of nodes each query visits during search and, using our node access distributions from above, assign each node in this sequence a value of 0 or 1 to indicate whether it is a hub node or not (we use a heuristic here that selects hub nodes based on p95/p99 of node access counts). 

We then save this sequence assignments as as numpy array. In order to reproduce this step run 

```shell 
./bin/docker-run.sh speed-test
```

Now you can generate similar plots by executing 

```shell
./bin/docker-run.sh plot-speed-test-results
```