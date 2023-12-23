## Near Neighbor Graph Reordering

This repository implements a graph near neighbor index and provides various ways to reorder the nodes in the graph to improve query latency. To reproduce our experiments, there are three tools that can be run:

- **construct** - creates a near neighbor index from a data file
- **reorder** - applies graph reordering to permute the node ordering of the index
- **query** - queries the index and computes recall and other performance metrics

The tools are largely self-documenting and will provide help if run without any command line arguments. Note that the reordering tools can generally run without needing access to the dataset, queries or distance metrics (unless profile-guided reordering is used).

1. `$ cd flatnav`
2. `chmod +x ./bin/build.sh`
2. `$ ./bin/build.sh `

This will build `flatnav` as a static library, but it will not build examples, unit tests and benchmarking
script. To see how to build other executables (ex. examples), run the build executable with the help 
option to see all valid options. 

```shell
> ./bin/build.sh -h 
```
It should print out a helpful message that looks like this
```shell

Usage ./build.sh [OPTIONS]

Available Options:
  -t, --tests:        Build tests
  -e, --examples:     Build examples
  -v, --verbose:      Make verbose
  -b, --benchmark:    Build benchmarks
  -h, --help:         Print this help message

Example Usage:
  ./build.sh -t -e -v
```

### Support for SIMD Extensions 

We currently support SIMD extensions for certain platforms as detailed below. 

| Operation | x86_64 | arm64v8 | Apple silicone |
|-----------|--------|---------|-----------------|
| FP32 Inner product |SSE, AVX, AVX512 | No SIMD support | No SIMD support |
| FP32 L2 distance |SSE, AVX, AVX512| No SIMD support | No SIMD support |



### System Requirements
Currently, we only support MacOS (and partially Ubuntu Linux). Before building Flatnav, you will need the following 

* C++17 capable compiler
* Git installation and cmake (version >= 3.14)
* If you are on MacOS, you will need to install Homebrew clang (necessary for OpenMP)

We provide some helpful scripts for installing the above in the [bin](/bin/) directory. 

### Datasets from ANN-Benchmarks

ANN-Benchmarks provides HDF5 files for a standard benchmark of near-neighbor datasets, queries and ground-truth results. To run on these datasets, we provide a set of tools to process numpy (NPY) files: construct_npy, reorder_npy and query_npy.

To generate these NPY files from the HDF5 files provided by ANN-benchmarks, you may use the Python script dump.py, as follows

```shell 
> python dump.py dataset.hdf5
```

Alternatively, you can use a helper script to download any ANN-benchmark script by running a command like 
this:

```shell
> ./bin/download_anns_datasets.sh glove-25-angular --normalize
```

For datasets that use the angular similarity measure, you will need the `--normalize` option so that the 
correct distance is computed. 


### Using Custom Datasets

The most straightforward way to include a new dataset for this evaluation is to put it into either the ANN-Benchmarks (NPY) format or to put it into the Big ANN-Benchmarks format. The NPY format requires a float32 2-D Numpy array for the train and test sets and an integer array for the ground truth. The Big ANN-Benchmarks format uses the following binary representation. For the train and test data, there is a 4-byte little-endian unsigned integer number of points followed by a 4-byte little-endian unsigned integer number of dimensions. This is followed by a flat list of `num_points * num_dimensions` values, where each value is a 32-bit float or an 8-bit integer (depending on the dataset type). The ground truth files consist of a 32-bit integer number of queries, followed by a 32-bit integer number of ground truth results for each query. This is followed by a flat list of ground truth results.


## Python Binding Instructions 
We also provide python bindings for a subset of index types. We've successfully built the bindings on Linux and MacOS, and if there is interest,
we can also support Windows. To generate the python bindings you will need a stable installation of [poetry](https://python-poetry.org/). 

Then, follow instructions [here](/flatnav_python/README.md) on how to build the library. There are also examples for how to use the library 
to build an index and run queries on top of it [here](/flatnav_python/test_index.py).




