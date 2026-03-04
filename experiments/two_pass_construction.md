# Two-Pass Construction Benchmarks

## BIGANN 100M (uint8, unquantized)

Native uint8 distance computation on the BIGANN dataset (vectors are already uint8).

```bash
make bigann-1b-anchor-uint8-bench
```

With docker, execute 
```bash 
./bin/docker-run.sh bigann-1b-anchor-uint8-bench
```

Or directly:

```bash
poetry run python evaluate_two_pass.py \
    --dataset /root/data/bigann-1b-euclidean/bigann_1b.train.npy \
    --queries /root/data/bigann-1b-euclidean/bigann_1b.test.npy \
    --gtruth /root/data/bigann-1b-euclidean/bigann_1b.gtruth.npy \
    --strategies baseline anchor \
    --distance-type l2 \
    --index-data-type uint8 \
    --num-threads 32 \
    --M-baseline 16 \
    --anchor-fraction 0.01 \
    --anchor-M 16 \
    --anchor-ef-construction 500 \
    --bulk-ef-construction 80 \
    --num-anchor-probes 10 \
    --output results/bigann-1b-anchor-uint8.json
```

## Yandex DEEP 100M (int8, scalar quantization)

Anchor construction with scalar quantization: float32 vectors are quantized to int8 on insertion. All distance computations use int8 SIMD kernels.

```bash
make yandex-deep-100m-sq-bench
```

With docker, execute 

```bash 
./bin/docker-run.sh yandex-deep-100m-sq-bench
```

Or directly:

```bash
poetry run python evaluate_two_pass.py \
    --dataset /root/data/yandex-deep-100m/yandex_100m.train.npy \
    --queries /root/data/yandex-deep-100m/yandex_100m.test.npy \
    --gtruth /root/data/yandex-deep-100m/yandex_100m.gtruth.npy \
    --strategies baseline sq \
    --distance-type l2 \
    --num-threads 32 \
    --M-baseline 16 \
    --anchor-fraction 0.01 \
    --anchor-M 16 \
    --anchor-ef-construction 500 \
    --sq-max-edges-per-node 16 \
    --sq-ef-construction 100 \
    --output results/yandex-deep-100m-sq.json
```
