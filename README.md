## FlatNav 

FlatNav is a fast and header-only graph-based index for Approximate Nearest Neighbor Search (ANNS). FlatNav is inspired by the influential [Hierarchical Navigable Small World (HNSW) index](https://github.com/nmslib/hnswlib), but with the hierarchical component removed. As detailed in our [research paper](https://arxiv.org/pdf/2412.01940), we found that FlatNav achieved identical performance to HNSW on high-dimensional datasets (dimensionality > 32) with approximately 38% less peak memory consumption and a simplified implementation. 

We hope to maintain this open source library as a resource for broader community. Please consider opening a Github Issue for bugs and feature requests, or get in touch with us directly for discussions.

### Reproducing Experimental Results from the Research Paper

In our associated [research paper](https://arxiv.org/pdf/2412.01940), we conduct a series of benchmarking experiments comparing FlatNav's non-hierarchical navigable small world graph index with HNSW. Ultimately, we find that in high-dimensional vector spaces, the hierarchical component of HNSW provides no discernible benefit in terms of search quality and performance compared to simply using a non-hierarchical NSW graph. To reproduce the benchmarking results reported in the paper, please see the [README file](https://github.com/BlaiseMuhirwa/flatnav/blob/main/experiments/README.md) located within the `experiments` directory of this repository. 

In addition to our benchmarking experiments, we also investigate *why* the hierarchical component of HNSW seems to not provide additional value on high-dimensional vector search workloads. In particular, we hypothesize that navigable small world graphs over high-dimensional metric spaces naturally form hubs. These hubs consist of a small subset of nodes that are highly connected to other points in the graph and thus facilitate fast traversal without the need for an explicit hierarchy. In our paper, we also perform a series of statistical tests to provide compelling evidence that our Hub Highway Hypothesis holds in practice. To reproduce these experimental findings, please see the [HUBNESS_EXPERIMENTS.md](https://github.com/BlaiseMuhirwa/flatnav/blob/main/experiments/HUBNESS_EXPERIMENTS.md) file. 

### Installation 
FlatNav is implemented in C++ with a complete Python extension with [cereal](https://uscilab.github.io/cereal/) as the only external dependency. This is a header-only library, so there is nothing to build. Currently, FlatNav is supported on x86-64 machines on Linux and MacOS (we can extend this to Windows and ARM platforms if there is sufficient interest).

#### Python Installation

For Python users, we recommend installing FlatNav via [pip](https://pypi.org/project/flatnav/)

```shell
pip install flatnav
```

Similarly, `flatnav` can be installed from source via [cibuildwheel](https://cibuildwheel.pypa.io/en/stable/), which 
builds cross-platform wheels. Follow the following steps

```shell
$ git clone https://github.com/BlaiseMuhirwa/flatnav.git --recurse-submodules
$ cd flatnav
$ make install-cibuildwheel

# This will build flatnav for the current version in your environment. If you want to build wheels 
# for all supported python versions (3.8 to 3.12), remove the --current-version flag.
$ ./cibuild.sh --current-version 3.12

$ pip install wheelhouse/flatnav*.whl --force-reinstall
```

Alternatively, if you just don't care about cross-platform builds and want to test it out quickly, you can simply run 

```shell
$ cd python-bindings
$ pip install .
```

#### C++ Installation

To get the C++ library working and run examples under the [tools](https://github.com/BlaiseMuhirwa/flatnav/blob/main/tools) directory, you will need

* C++17 compiler with OpenMP support (version >= 2.0)
* CMake (version >= 3.14)

We provide some helpful scripts for installing the above in the [bin](https://github.com/BlaiseMuhirwa/flatnav/tree/main/bin) directory. 

To generate the library with CMake and compile examples, run 

```shell
$ git clone https://github.com/BlaiseMuhirwa/flatnav.git --recurse-submodules
$ cd flatnav
$ ./bin/build.sh -e
```

You can get all options available with the `build.sh` script by passing it the `-h` argument.

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

### Getting Started in Python

Currently, we support Python wheels for versions 3.8 through 3.12 on x86_64 architectures (Intel, AMD and MacOS).

Once you have the python library installed and you have a dataset you want to index as a numpy array, you can construct the index as shown below. This will allocate memory and create a directed graph with vectors as nodes.

```python
import numpy as np
import flatnav
from flatnav.data_type import DataType 

# Get your numpy-formatted dataset.
dataset_size = 1_000_000
dataset_dimension = 128
dataset_to_index = np.random.randn(dataset_size, dataset_dimension)

# Define index construction parameters.
distance_type = "l2"
max_edges_per_node = 32
ef_construction = 100
num_build_threads = 16

# Create index configuration and pre-allocate memory
index = flatnav.index.create(
    distance_type=distance_type,
    index_data_type=DataType.float32,
    dim=dataset_dimension,
    dataset_size=dataset_size,
    max_edges_per_node=max_edges_per_node,
    verbose=True,
    collect_stats=True,
)
index.set_num_threads(num_build_threads)

# Now index the dataset 
index.add(data=dataset_to_index, ef_construction=ef_construction)
```

Note that we specified `DataType.float32` to indicate that we want to build an index with vectors represented with `float` type. If you want to use a different precision, such as `uint8_t` or `int8_t` (which are the only other ones currently supported), you can use `DataType.uint8` or `DataType.int8`.
The distance type can either be `l2` or `angular`. The `collect_stats` flag will record the number of distance evaluations.

To query the index we just created by generating IID vectors from the standard normal distribution, we do it as follows 

```python

# Set query-time parameters 
k = 100
ef_search = 100

# Run k-NN query with a single thread.
index.set_num_threads(1)

queries = np.random.randn(1000, dataset_to_index.shape[1])
for query in queries:
  distances, indices = index.search_single(
    query=query,
    ef_search=ef_search,
    K=k,
  )

```

You can parallelize the search by setting the number of threads to a desired number and using a different API that also returns the exact same results as `search_single`.

```python
index.set_num_threads(16)
distances, indices = index.search(queries=queries, ef_search=ef_search, K=k)
```

### Getting Started in C++

As mentioned earlier, there is nothing to build since this is header-only. We will translate the above Python code in C++ to illustrate how to use the C++ API. 

```c++
#include <cstdint>
#include <flatnav/index/Index.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/distances/DistanceInterface.h>

template <typename dist_t>
void run_knn_search(Index<dist_t, int>>* index, float *queries, int* gtruth, 
        int ef_search, int K, int num_queries, int num_gtruth, int dim) {

  float mean_recall = 0;
  for (int i = 0; i < num_queries; i++) {
    float *q = queries + dim * i;
    int *g = gtruth + num_gtruth * i;
    std::vector<std::pair<float, int>> result =
        index->search(q, K, ef_search);

    float recall = 0;
    for (int j = 0; j < K; j++) {
      for (int l = 0; l < K; l++) {
        if (result[j].second == g[l]) {
          recall = recall + 1;
        }
      }
    }
    recall = recall / K;
    mean_recall = mean_recall + recall;
  }
}


int main(int argc, char** argv) {
  uint32_t dataset_size = 1000000;
  uint32_t dataset_dimension = 128;

  // We skip the random data generation, but you can do that with std::mt19937, std::random_device 
  // and std::normal_distribution
  // std::vector<float> dataset_to_index; 

  uint32_t max_edges_per_node = 32;
  uint32_t ef_construction = 100;

  // Create an index with l2 distance 
  auto distance = SquaredL2Distance<>::create(dataset_dimension);
  auto* index = new Index<SquaredL2Distance<DataType::float32>>, int>(
      /* dist = */ std::move(distance), /* dataset_size = */ dataset_size,
      /* max_edges_per_node = */ max_edges_per_node);

  index->setNumThreads(build_num_threads);

  std::vector<int> labels(dataset_size);
  std::iota(labels.begin(), labels.end(), 0);
  index->template addBatch<float>(/* data = */ (void *)dataset_to_index,
                                  /* labels = */ labels,
                                  /* ef_construction */ ef_construction);

  // Now query the index and compute the recall 
  // We assume you have a ground truth (int*) array and a queries (float*) array
  uint32_t ef_search = 100;
  uint32_t k = 100;
  uint32_t num_queries = 1000;
  uint32_t num_gtruth = 1000;
  
  // Query the index and compute the recall. 
  run_knn_search(index, queries, gtruth, ef_search, k, num_queries, num_gtruth, dataset_dimension);
}

``` 

### Experimental API and Future Extensions 

You can find the current work under development under the [development-features](https://github.com/BlaiseMuhirwa/flatnav/blob/main/development-features) directory. 
While some of these features may be usable, they are not guarranteed to be stable. Stable features will be expected to be part of the PyPI releases. 
The most notable on-going extension that's under development is product quantization.

## Citation
If you find this library useful, please consider citing our associated paper:

```
@article{munyampirwa2024down,
  title={Down with the Hierarchy: The'H'in HNSW Stands for" Hubs"},
  author={Munyampirwa, Blaise and Lakshman, Vihan and Coleman, Benjamin},
  journal={arXiv preprint arXiv:2412.01940},
  year={2024}
}
```
