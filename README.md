## FlatNav 

FlatNav is a fast and header-only graph-based index for Approximate Nearest Neighbor Search (ANNS).
Thanks to various graph re-ordering implementation techniques from [Coleman et al.](https://arxiv.org/pdf/2104.03221.pdf), optimization using SIMD intrinsics, and the design that leverages a single-layered 
in-memory index, FlatNav provides billion-scale vector search with high recall, SoTA query latency and 
significant memory savings. 

### Near Neighbor Graph Reordering

We provide various ways to re-order nodes in the graph to improve query latency. To reproduce our experiments, there are three tools that can be run:

- **construct** - creates a near neighbor index from a data file
- **reorder** - applies graph reordering to permute the node ordering of the index
- **query** - queries the index and computes recall and other performance metrics

The tools are largely self-documenting and will provide help if run without any command line arguments. Note that the reordering tools can generally run without needing access to the dataset, queries or distance metrics (unless profile-guided reordering is used).

For python examples that involve re-ordering, you can run [run-benchmark.py](https://github.com/BlaiseMuhirwa/flatnav-experimental/blob/main/experiments/run-benchmark.py) with `--reodering-strategies gorder rcm`. See example runs in the [Makefile](https://github.com/BlaiseMuhirwa/flatnav-experimental/blob/main/experiments/Makefile).

### Installation 
FlatNav is implemented in C++ with a complete Python extension with [cereal](https://uscilab.github.io/cereal/) as the only external dependency. The C++ library can be built from source using CMake. 

FlatNav is supported on x86-64 machines on linux and MacOS (we can extend this to windows if there is sufficient interest). To build from source
you will need

* C++17 compiler with OpenMP support (version >= 2.0)
* CMake (version >= 3.14)

We provide some helpful scripts for installing the above in the [bin](https://github.com/BlaiseMuhirwa/flatnav-experimental/tree/main/bin) directory. 

To build the library with CMake, run 

```shell
$ git clone https://github.com/BlaiseMuhirwa/flatnav-experimental.git --recurse-submodules
$ cd flatnav-experimental
$ ./bin/build.sh -h
```

This will display all available build options:

```shell
Usage ./build.sh [OPTIONS]

Available Options:
  -t, --tests:                    Build tests
  -e, --examples:                 Build examples
  -v, --verbose:                  Make verbose
  -b, --benchmark:                Build benchmarks
  -bt, --build_type:              Build type (Debug, Release, RelWithDebInfo, MinSizeRel)
  -nmv, --no_simd_vectorization:Disable SIMD instructions
  -h, --help:                     Print this help message

Example Usage:
  ./build.sh -t -e -v
```

To build the Python bindings, follow instructions [here](https://github.com/BlaiseMuhirwa/flatnav-experimental/blob/main/flatnav_python/README.md). There are also examples for how to use the library to build an index and run queries on top of it [here](https://github.com/BlaiseMuhirwa/flatnav-experimental/blob/main/flatnav_python/unit_tests/test_index.py).

### Support for SIMD Extensions 

We currently support SIMD extensions for certain platforms as detailed below. 

| Operation | x86_64 | arm64v8 | Apple silicone |
|-----------|--------|---------|-----------------|
| FP32 Inner product |SSE, AVX, AVX512 | No SIMD support | No SIMD support |
| FP32 L2 distance |SSE, AVX, AVX512| No SIMD support | No SIMD support |


### Datasets from ANN-Benchmarks

ANN-Benchmarks provides HDF5 files for a standard benchmark of near-neighbor datasets, queries and ground-truth results. To run on these datasets, we provide a set of tools to process numpy (NPY) files: construct_npy, reorder_npy and query_npy.

To generate the [ANNS benchmark datasets](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file#data-sets), run the following script

```shell
$ ./bin/download_anns_datasets.sh <dataset-name> [--normalize]
```

For datasets that use the angular/cosine similarity, you will need to use `--normalize` option so that the dataset is normalized. 
Available dataset names include:

* mnist-784-euclidean
* sift-128-euclidean
* glove-25-angular
* glove-50-angular
* glove-100-angular
* glove-200-angular
* deep-image-96-angular
* gist-960-euclidean
* nytimes-256-angular



### Using Custom Datasets

The most straightforward way to include a new dataset for this evaluation is to put it into either the ANN-Benchmarks (NPY) format or to put it into the Big ANN-Benchmarks format. The NPY format requires a float32 2-D Numpy array for the train and test sets and an integer array for the ground truth. The Big ANN-Benchmarks format uses the following binary representation. For the train and test data, there is a 4-byte little-endian unsigned integer number of points followed by a 4-byte little-endian unsigned integer number of dimensions. This is followed by a flat list of `num_points * num_dimensions` values, where each value is a 32-bit float or an 8-bit integer (depending on the dataset type). The ground truth files consist of a 32-bit integer number of queries, followed by a 32-bit integer number of ground truth results for each query. This is followed by a flat list of ground truth results.