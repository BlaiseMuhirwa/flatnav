## Datasets from ANN-Benchmarks

ANN-Benchmarks provide HDF5 files for a standard benchmark of near-neighbor datasets, queries and ground-truth results. To index any of these datasets you can use the `construct_npy.cpp` and `query_npy.cpp` files linked above.

To generate the [ANNS benchmark datasets](https://github.com/erikbern/ann-benchmarks?tab=readme-ov-file#data-sets), run the following script

```shell
$ ./bin/download_anns_datasets.sh <dataset-name> [--normalize]
```

For datasets that use the angular/cosine similarity, you will need to use `--normalize` option so that the distances are computed correctly. 

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

## Instructions for using the Experiment Runner

We currently have two different ways to build the Python library and use the
experiment runner:

1. Building the python library directly from source and running the `run-benchmark.py`
2. Using docker to build the library. 

We highly recommend using the second option since it's good for reproducibility and you 
won't need to personally manage a poetry installation. If you do insist on not using docker
for whatever reason, you will need to first install [poetry](https://python-poetry.org/) and follow [this](../flatnav_python/README.md) guide for building the library. 

### Running experiments from a docker image 

Here is the workflow for running experiments: 

1. Make changes to the run-benchmark.py script per your needs.
2. Add a make target in the [Makefile](/experiments/Makefile) if needed or edit an existing target
3. Build a docker image to reflect these changes

The docker image that we provide packages the C++ source code and uses that to build the python library
and the experiment runner script. 

To build the docker image, run the following commands:

```shell
# Go to the root level
> cd ..
> ./bin/docker-build.sh 
```

If you don't provide a target, this will build the image and print the available targets in the [Makefile](/experiments/Makefile) like so:

```
Usage: make [target]
Targets:
  setup: install all dependencies including flatnav
  install-flatnav: install flatnav
  install-hnswlib: install hnswlib
  cleanup: remove hnswlib-original
  generate-wheel: generate wheel for flatnav
  yandex-deep-bench: run yandex-deep benchmark
  sift-bench: run sift benchmark
```

You can specify whatever target you want by running, for example

```
./bin/docker-test.sh sift-bench-flatnav
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

You may want to log the experiment logs to a file on disk. You can do so by running 
```
> ./bin/docker-test.sh sift-bench > logs.txt 2>& 1
```


### Pushing Index Snapshots to S3

This requires the following

* Saving the AWS credentials to a `~/.aws/credentials` file. Let's put these under the `s3-bucket-reader-writer` profile to 
be consistent with what the [.env-vars](bin/.env-vars) file expects as environment variable. The `~/.aws` will be mounted as a volume 
inside docker as a read-only directory. `boto3` will use these credentials to authenticate with AWS. 
* Turn on S3 push feature. You can do this by setting `DISABLE_PUSH_TO_S3` to `0` in the [.env-vars](/bin/.env-vars) file. 

Then, run `./bin/docker-run.sh <make-target>` as usual. We run two processes concurrently using supervisor. One process
will run the benchmark while the other one continuously looks for indexes inside the container, retrieves the most
recent snapshot, and pushes it to the pre-defined S3 bucket defined in the .env file. 

To see the logs from this second process, run 

```shell
docker exec -it benchmark-runner tail -f /var/log/cron_stderr.log
```

or replace `/var/log/cron_stderr.log` with `/var/log/cron_stdout.log`. 

This process runs like a "cronjob" that polls for HNSW indexes every 30 seconds. 



