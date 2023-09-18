## Benchmark Flatnav against ANNS Datasets 

### Preparing Datasets 

Before trying to run benchmarks, make sure you have datasets ready locally. 
There are helper scripts inside the `bin` directory that should be useful. 
One script in particular that you should run is [dowload_anns_datasets.sh](/bin/download_anns_datasets.sh).

To see the available options, run 
```shell
> ./bin/download_anns_datasets.sh -h
```

It will display a message like this, indicating how to use it to download pretty
much any dataset from the [ANNS](https://github.com/erikbern/ann-benchmarks) website
as HDF5-formatted, and then parses it into training data, testing and query datasets. 

```python
Usage: ./download_anns_datasets.sh <dataset> [--normalize]

Available datasets:
mnist-784-euclidean sift-128-euclidean 
glove-25-angular glove-100-angular 
glove-50-angular glove-200-angular 
deep-image-96-angular gist-960-euclidean 
nytimes-256-angular

Example Usage:
  ./download_anns_datasets.sh mnist-784-euclidean
  ./download_anns_datasets.sh glove-25-angular --normalize
```


### Running Benchmarks 

This directory contains the following files that are useful for benchmarking Flatnav against [ANNS Datasets](https://github.com/erikbern/ann-benchmarks).

They include the following:
* `config.yaml`: This is a YAML specification that includes a list of benchmark datasets
and the desired parameters for running benchmarks, such as `ef_construction`, `ef_search`,
`dataset_size` etc. 

* `config_parser.h`: It does what the name suggests (i.e., parses the YAML config).
* `runner.cpp`: Runs benchmarks against all specified datasets. This utilizes [google benchmark](https://github.com/google/benchmark) for now. I noticed that google benchmark is pretty limited, but until we find a better alternative this will do for now. 

To run benchmarks, do the following; 
```shell
# This will build the library and the runner.cpp executable 
> bin/build.sh -b # Equivalently bin/build.sh --benchmark
> ./build/run_benchmark 
```

`NOTE`: `./build/run_benchmark` can be given a subset of datasets to run benchmarks on. 
This is desirable since `config.yaml` contains many datasets and one might not want to run
all of them at once. To benchmark Flatnav against just a subset of datasets, pass them to the 
script as comma-separated strings like this:

```shell
> ./build/run_benchmark --datasets mnist-784-euclidean,sift-128-euclidean
```

And of course to see how to use this executable, you can just run 
```shell
> ./build/run_benchmark --help
```