# Two-Pass Construction Benchmarks

## Running with Docker

All make targets run inside Docker via `./bin/docker-run.sh <target>`. Set `DATA_DIR` to the host directory containing your datasets -- it gets mounted to `/root/data` inside the container.

```bash
export DATA_DIR=/path/to/your/data
./bin/docker-run.sh sift-anchor-sq-bench
```

Any output saved to `/root/data/...` inside the container will appear under `$DATA_DIR/...` on the host. 

## Scalar Quantization (SQ)

SQ maps each float32 dimension to int8 using per-dimension min/max uniform quantization. A training pass computes per-dimension min/max values, then each float is transformed as:

$$
q_d = \text{round}\!\left(\frac{x_d - \min_d}{\max_d - \min_d} \times 255\right) - 128
$$

where $q_d \in [-128, 127]$ is the stored int8 value. This reduces storage from 4 bytes to 1 byte per dimension (4x), and all distance computations use int8 SIMD kernels.

Use `--use-sq-quantization` to enable SQ for both baseline and anchor strategies. Use `--sq-quant-max-train-samples N` to subsample the set of vectors to use for computing the per-dimension min and max. 

## Data Prep

Each benchmark expects three `.npy` files: train vectors, query vectors, and ground truth indices. Use `bin/download_bigann_datasets.sh` to download and convert datasets automatically.

### Downloading BIGANN 1B

```bash
./bin/download_bigann_datasets.sh -d /path/to/your/data --dataset bigann
```

This downloads the full 1B dataset and produces:

```
/path/to/your/data/bigann/train.npy          # 1B train vectors (uint8)
/path/to/your/data/bigann/queries.npy         # 10K query vectors
/path/to/your/data/bigann/ground_truth_1b.npy # 1B ground truth
/path/to/your/data/bigann/ground_truth_100m.npy
/path/to/your/data/bigann/ground_truth_10m.npy
```

### Downloading Yandex DEEP 1B

```bash
./bin/download_bigann_datasets.sh -d /path/to/your/data --dataset deep
```

This produces:

```
/path/to/your/data/deep/train.npy          # 1B train vectors (float32)
/path/to/your/data/deep/queries.npy         # 10K query vectors
/path/to/your/data/deep/ground_truth_1b.npy # 1B ground truth
/path/to/your/data/deep/ground_truth_100m.npy
/path/to/your/data/deep/ground_truth_10m.npy
```

### Downloading a smaller subset

Use `--chunk-size` to download the full binary but only convert a subset of train vectors:

```bash
./bin/download_bigann_datasets.sh -d /path/to/your/data --dataset bigann --chunk-size 100000000
```

This saves `train_100m.npy` instead of the full `train.npy`.

### Running benchmarks

Set `DATA_DIR` to the download directory so Docker mounts it to `/root/data`:

```bash
export DATA_DIR=/path/to/your/data
```

The container mounts `$DATA_DIR` to `/root/data`, so `/root/data/bigann/train.npy` inside the container maps to `$DATA_DIR/bigann/train.npy` on the host.

## Validation with SIFT 1M

To check that quantizing with int8 is doing the right thing, you can run simple validation via 

```bash 
# If you don't do "export DATA_DIR=/some/path/to/dataset", this will be stored under ./data
./bin/download_ann_benchmarks_datasets.sh sift-128-euclidean
./bin/docker-run.sh sift-anchor-sq-bench
```

NOTE: You typically shouldn't need to int8-quantize SIFT since the dataset is in uint8 format, but this should serve as a pretty simple check to make sure the setup with docker works. 

## BIGANN 1B (uint8, unquantized)

Native uint8 distance computation on the BIGANN dataset (vectors are already uint8).

```bash
./bin/docker-run.sh bigann-1b-anchor-uint8-bench
```

Parameters:

* `--strategies baseline anchor` -- runs both single-pass baseline and anchor construction
* `--index-data-type uint8` -- stores vectors as native uint8 (no quantization needed, BIGANN is already uint8)
* `--num-threads 32` -- parallel construction with 32 threads. If you want to use more or less threads, you can modify this from the `experiments/Makefile`. 
* `--M-baseline 16` -- 16 edges per node for baseline
* `--anchor-fraction 0.01` -- 1% of vectors used as anchors
* `--anchor-M 16` -- 16 edges per node in the graph constructed with the anchor method
* `--anchor-ef-construction 500` -- high ef for building the anchor subgraph
* `--bulk-ef-construction 80` -- ef for inserting non-anchor vectors
* `--num-anchor-probes 50` -- number of anchor nodes probed to find entry points during bulk insertion (the second pass). We usually refer to this as `num_initialization` is the standard flatnav API. 


## Yandex DEEP 1B (float32, scalar quantization)

Anchor construction with scalar quantization: float32 vectors are quantized to int8 on insertion. 

```bash
./bin/docker-run.sh yandex-deep-1b-sq-bench
```

Parameters:

* `--strategies baseline anchor` -- runs both single-pass baseline and anchor construction (both with SQ)
* `--num-threads 32` -- parallel construction with 32 threads
* `--M-baseline 16` -- 16 edges per node for baseline
* `--anchor-fraction 0.01` -- 1% of vectors used as anchors
* `--anchor-M 16` -- 16 edges per node in the anchor/bulk graph
* `--anchor-ef-construction 500` -- high ef for building the anchor subgraph
* `--bulk-ef-construction 100` -- ef for inserting non-anchor vectors
* `--num-anchor-probes 50` -- number of anchor nodes probed to find entry points during bulk insertion
* `--use-sq-quantization` -- enable int8 scalar quantization for both strategies
* `--sq-quant-max-train-samples 100000000` -- subsample 100M vectors for min/max derivation

## Other available benchmarks

BIGANN 100M and Yandex DEEP 100M anchor benchmarks (without scalar quantization):

```bash
./bin/docker-run.sh bigann-100m-anchor-bench
./bin/docker-run.sh yandex-deep-100m-anchor-bench
```
