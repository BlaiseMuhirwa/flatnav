# This file contains the benchmarking script configurations for various near neighbor datasets
# The main configurable parameter is the type of index: --index-type (hnsw, flatnav)
# We use dataset-name for the name of the plots.

profile:
	perf stat -e cache-references,cache-misses,LLC-load-misses,LLC-loads,L1-dcache-load-misses,L1-dcache-loads,dTLB-load-misses,dTLB-loads,instructions,cycles -- sleep 0.1

################################
# Begin 100-Million Scale Benchmarks
# ################################
bigann-100m-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name bigann-100m \
		--dataset /root/data/bigann/train_100m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_100m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename bigann-100m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

bigann-100m-two-pass-bench-flatnav:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/bigann-100m-euclidean/bigann_100m.train.npy \
		--queries /root/data/bigann-100m-euclidean/bigann_100m.test.npy \
		--gtruth /root/data/bigann-100m-euclidean/bigann_100m.gtruth.npy \
		--strategies re_prune \
		--num-threads 32 \
		--M-baseline 16 \
		--M-pass1 8 \
		--M-pass2 8 \
		--ef-construction-pass1 50 \
		--ef-construction-pass2 50 \
		--output /root/data/two-pass-construction/bigann-100m-two-pass-bench-flatnav.json

bigann-100m-anchor-bench:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/bigann-100m-euclidean/bigann_100m.train.npy \
		--queries /root/data/bigann-100m-euclidean/bigann_100m.test.npy \
		--gtruth /root/data/bigann-100m-euclidean/bigann_100m.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/bigann-100m-anchor-bench.json \
		--perf-output /root/data/anchor-construction/bigann-100m-anchor-perf.json

bigann-1b-anchor-uint8-bench:
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
		--output /root/data/anchor-construction/bigann-1b-anchor-uint8-bench.json \
		--perf-output /root/data/anchor-construction/bigann-1b-anchor-uint8-perf.json

bigann-100m-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name bigann-100m \
		--dataset /root/data/bigann/train_100m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_100m.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

yandex-deep-100m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-100m \
		--dataset /root/data/deep/train_100m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_100m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename yandex-deep-100m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


yandex-deep-100m-anchor-bench:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/yandex-deep-100m/yandex_100m.train.npy \
		--queries /root/data/yandex-deep-100m/yandex_100m.test.npy \
		--gtruth /root/data/yandex-deep-100m/yandex_100m.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/yandex-deep-100m-anchor-bench.json \
		--perf-output /root/data/anchor-construction/yandex-deep-100m-anchor-perf.json

# Controlled experiment: isolate cache locality on YANDEX DEEP-100M.
# Both methods use IDENTICAL probe count (100) and ef_construction (100).
# The ONLY difference is WHERE the 100 probes land:
#   baseline: uniformly across all 100M nodes (~55 GB span, far exceeds L3)
#   anchor:   uniformly across first 1M nodes (~550 MB span, fits in L3)
# At this scale, DRAM fills should diverge — baseline probes go to DRAM,
# anchor probes stay in L3.
yandex-deep-100m-sq-bench:
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
		--output /root/data/sq-construction/yandex-deep-100m-sq-bench.json \
		--perf-output /root/data/sq-construction/yandex-deep-100m-sq-perf.json

yandex-deep-100m-anchor-controlled:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/yandex-deep-100m/yandex_100m.train.npy \
		--queries /root/data/yandex-deep-100m/yandex_100m.test.npy \
		--gtruth /root/data/yandex-deep-100m/yandex_100m.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--ef-construction-baseline 100 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 100 \
		--num-anchor-probes 100 \
		--output /root/data/anchor-construction/yandex-deep-100m-anchor-controlled.json \
		--perf-output /root/data/anchor-construction/yandex-deep-100m-anchor-controlled-perf.json

yandex-deep-100m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-100m \
		--dataset /root/data/deep/train_100m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_100m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


spacev-100m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name spacev-100m \
		--dataset /root/data/msspacev/train_100m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_100m.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename spacev-100m.mtx \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

spacev-100m-anchor-bench:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/SPTAG/datasets/SPACEV1B/collection.npy \
		--queries /root/data/SPTAG/datasets/SPACEV1B/spacev_100m.test.npy \
		--gtruth /root/data/SPTAG/datasets/SPACEV1B/spacev_100m.gtruth.npy \
		--range 0 100000000 \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/spacev-100m-anchor-bench.json \
		--perf-output /root/data/anchor-construction/spacev-100m-anchor-perf.json

spacev-100m-anchor-bench-with-data-loader:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/SPTAG/datasets/SPACEV1B/vectors.bin \
		--queries /root/data/SPTAG/datasets/SPACEV1B/query.bin \
		--gtruth /root/data/SPTAG/datasets/SPACEV1B/truth.bin \
		--range 0 100000000 \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/spacev-100m-anchor-bench.json \
		--perf-output /root/data/anchor-construction/spacev-100m-anchor-dataloader-perf.json


spacev-100m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name spacev-100m \
		--dataset /root/data/msspacev/train_100m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_100m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


tti-100m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name tti-100m \
		--dataset /root/data/text2image/train_100m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_100m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename tti-100m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1

tti-100m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name tti-100m \
		--dataset /root/data/text2image/train_100m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_100m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1


tti-100m-anchor-bench:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/SPTAG/datasets/SPACEV1B/spacev_100m.train.npy \
		--queries /root/data/SPTAG/datasets/SPACEV1B/spacev_100m.test.npy \
		--gtruth /root/data/SPTAG/datasets/SPACEV1B/spacev_100m.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/spacev-100m-anchor-bench.json \
		--perf-output /root/data/anchor-construction/tti-100m-anchor-perf.json

################################
# End 100-Million Scale Benchmarks
# # ################################

################################
# Begin 10-Million Scale Benchmarks
# # ################################
bigann-10m-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name bigann-10m \
		--dataset /root/data/bigann/train_10m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_10m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename bigann-10m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


bigann-10m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name bigann-10m \
		--dataset /root/data/bigann/train_10m.npy \
		--queries /root/data/bigann/queries.npy \
		--gtruth /root/data/bigann/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1

yandex-deep-10m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-10m \
		--dataset /root/data/deep/train_10m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_10m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename yandex-deep-10m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


yandex-deep-10m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name yandex-deep-10m \
		--dataset /root/data/deep/train_10m.npy \
		--queries /root/data/deep/queries.npy \
		--gtruth /root/data/deep/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


spacev-10m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name spacev-10m \
		--dataset /root/data/msspacev/train_10m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_10m.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename spacev-10m.mtx \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


spacev-10m-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name spacev-10m \
		--dataset /root/data/msspacev/train_10m.npy \
		--queries /root/data/msspacev/queries.npy \
		--gtruth /root/data/msspacev/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 32 \
		--num-search-threads 1


tti-10m-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name tti-10m \
		--dataset /root/data/text2image/train_10m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_10m.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename tti-10m.mtx \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1

tti-10m--bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name tti-10m \
		--dataset /root/data/text2image/train_10m.npy \
		--queries /root/data/text2image/queries.npy \
		--gtruth /root/data/text2image/ground_truth_10m.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric angular \
		--num-build-threads 32 \
		--num-search-threads 1

################################
# End 10-Million Scale Benchmarks
# # ################################

################################
# Begin ANN Benchmarks
# Ref: https://ann-benchmarks.com/
################################

deep-image-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset /root/data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries /root/data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth /root/data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--index-type flatnav \
		--hnsw-base-layer-filename deep-image.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1

deep-image-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset /root/data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries /root/data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth /root/data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
        --num-search-threads 1

mnist-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name mnist-784 \
		--dataset /root/data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries /root/data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth /root/data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename mnist.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

mnist-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name mnist-784 \
		--dataset /root/data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries /root/data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth /root/data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

gist-bench-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename gist.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


gist-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


sift-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1

sift-bench-flatnav-cachegrind:
	valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.baseline.out \
		poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 100 \
		--ef-search 100 \
		--metric l2 \
		--num-build-threads 1 \
		--num-search-threads 1

sift-bench-two-pass-bench-flatnav:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--strategies re_prune \
		--num-threads 8 \
		--M-baseline 16 \
		--M-pass1 8 \
		--M-pass2 8 \
		--ef-construction-pass1 50 \
		--ef-construction-pass2 50 \
		--output /root/data/two-pass-construction/sift-bench-two-pass-bench-flatnav.json

# Controlled experiment: isolate cache locality effect.
# Both methods use IDENTICAL probe count (100) and ef_construction (100).
# The ONLY difference is WHERE the 100 probes land:
#   baseline: uniformly across all N nodes (scattered memory accesses)
#   anchor:   uniformly across first 1% of nodes (compact, contiguous)
# Any difference in perf counters is purely due to cache locality.
sift-anchor-controlled:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 8 \
		--M-baseline 16 \
		--ef-construction-baseline 100 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 100 \
		--num-anchor-probes 100 \
		--output /root/data/anchor-construction/sift-anchor-controlled.json \
		--perf-output /root/data/anchor-construction/sift-anchor-controlled-perf.json

sift-bench-two-pass-bench-flatnav-cachegrind:
	valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.two_pass.out \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--strategies re_prune \
		--num-threads 1 \
		--M-baseline 16 \
		--M-pass1 8 \
		--M-pass2 8 \
		--ef-construction-pass1 50 \
		--ef-construction-pass2 50 \
		--output /root/data/two-pass-construction/sift-bench-two-pass-bench-flatnav.json

# Perf-based cache analysis (requires --privileged container)
# Output saved to /root/metrics/ (mount a volume there)
# To see available events on your CPU, run: perf list cache
sift-bench-flatnav-perf:
	perf stat -o /root/metrics/perf-baseline.txt \
		-e cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-prefetches,dTLB-load-misses,dTLB-loads,instructions,cycles \
		poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--num-node-links 16 \
		--ef-construction 100 \
		--ef-search 100 \
		--metric l2 \
		--num-build-threads 8 \
		--num-search-threads 1

perf-listing: 
	perf list cache

sift-bench-two-pass-bench-flatnav-perf:
	perf stat -o /root/metrics/perf-two-pass.txt \
		-e cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads,L1-dcache-prefetches,dTLB-load-misses,dTLB-loads,instructions,cycles \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--strategies re_prune \
		--num-threads 8 \
		--M-baseline 16 \
		--M-pass1 8 \
		--M-pass2 8 \
		--ef-construction-pass1 50 \
		--ef-construction-pass2 50 \
		--output /root/data/two-pass-construction/sift-bench-two-pass-bench-flatnav.json

################################
# Anchor Construction Perf Analysis
# Run with: ./bin/docker-run.sh --privileged <target>
# These targets collect comprehensive cache/memory statistics to validate
# the hypothesis that anchor nodes at the start of the index improve cache locality.
################################

# Common perf events for cache analysis:
# - cache-misses/cache-references: Overall cache miss rate
# - L1-dcache-load-misses/loads: L1 data cache (fastest, 1-4 cycles)
# - LLC-load-misses/LLC-loads: Last Level Cache misses go to DRAM (~100+ cycles)
# - dTLB-load-misses/loads: TLB misses trigger page table walks
# - page-faults: Page faults (major = disk, minor = already in memory)
# - instructions/cycles: For computing IPC (instructions per cycle)

PERF_EVENTS_CACHE = cache-misses,cache-references,L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads,dTLB-load-misses,dTLB-loads,page-faults,instructions,cycles

# SIFT-scale anchor perf (1M vectors - quick validation)
sift-anchor-bench-perf:
	@echo "=== Running baseline construction with perf ==="
	perf stat -o /root/metrics/perf-anchor-baseline.txt \
		-e $(PERF_EVENTS_CACHE) \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--strategies baseline \
		--num-threads 1 \
		--M-baseline 16 \
		--output /root/data/anchor-construction/sift-anchor-baseline-perf.json
	@echo "=== Running anchor construction with perf ==="
	perf stat -o /root/metrics/perf-anchor-anchor.txt \
		-e $(PERF_EVENTS_CACHE) \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--strategies anchor \
		--num-threads 1 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/sift-anchor-anchor-perf.json
	@echo "=== Perf results saved to /root/metrics/perf-anchor-*.txt ==="

# BigANN 100M anchor perf (100M vectors - full scale validation)
# Use single thread to get cleaner per-operation measurements
bigann-100m-anchor-bench-perf:
	@echo "=== Running baseline construction with perf (100M) ==="
	perf stat -o /root/metrics/perf-bigann100m-baseline.txt \
		-e $(PERF_EVENTS_CACHE) \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/bigann-100m-euclidean/bigann_100m.train.npy \
		--queries /root/data/bigann-100m-euclidean/bigann_100m.test.npy \
		--gtruth /root/data/bigann-100m-euclidean/bigann_100m.gtruth.npy \
		--strategies baseline \
		--num-threads 1 \
		--M-baseline 16 \
		--output /root/data/anchor-construction/bigann-100m-baseline-perf.json
	@echo "=== Running anchor construction with perf (100M) ==="
	perf stat -o /root/metrics/perf-bigann100m-anchor.txt \
		-e $(PERF_EVENTS_CACHE) \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/bigann-100m-euclidean/bigann_100m.train.npy \
		--queries /root/data/bigann-100m-euclidean/bigann_100m.test.npy \
		--gtruth /root/data/bigann-100m-euclidean/bigann_100m.gtruth.npy \
		--strategies anchor \
		--num-threads 1 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/bigann-100m-anchor-perf.json
	@echo "=== Perf results saved to /root/metrics/perf-bigann100m-*.txt ==="

# Parallel construction perf (to see if cache benefits scale with threads)
bigann-100m-anchor-bench-perf-parallel:
	@echo "=== Running baseline construction with perf (100M, 32 threads) ==="
	perf stat -o /root/metrics/perf-bigann100m-baseline-parallel.txt \
		-e $(PERF_EVENTS_CACHE) \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/bigann-100m-euclidean/bigann_100m.train.npy \
		--queries /root/data/bigann-100m-euclidean/bigann_100m.test.npy \
		--gtruth /root/data/bigann-100m-euclidean/bigann_100m.gtruth.npy \
		--strategies baseline \
		--num-threads 32 \
		--M-baseline 16 \
		--output /root/data/anchor-construction/bigann-100m-baseline-parallel-perf.json
	@echo "=== Running anchor construction with perf (100M, 32 threads) ==="
	perf stat -o /root/metrics/perf-bigann100m-anchor-parallel.txt \
		-e $(PERF_EVENTS_CACHE) \
		poetry run python evaluate_two_pass.py \
		--dataset /root/data/bigann-100m-euclidean/bigann_100m.train.npy \
		--queries /root/data/bigann-100m-euclidean/bigann_100m.test.npy \
		--gtruth /root/data/bigann-100m-euclidean/bigann_100m.gtruth.npy \
		--strategies anchor \
		--num-threads 32 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--bulk-ef-construction 80 \
		--num-anchor-probes 10 \
		--output /root/data/anchor-construction/bigann-100m-anchor-parallel-perf.json
	@echo "=== Perf results saved to /root/metrics/perf-bigann100m-*-parallel.txt ==="

# Helper to print a summary comparison of perf results
perf-anchor-summary:
	@echo "=== Cache Statistics Summary ==="
	@echo ""
	@echo "--- Baseline ---"
	@cat /root/metrics/perf-anchor-baseline.txt 2>/dev/null || cat /root/metrics/perf-bigann100m-baseline.txt 2>/dev/null || echo "No baseline perf data found"
	@echo ""
	@echo "--- Anchor ---"
	@cat /root/metrics/perf-anchor-anchor.txt 2>/dev/null || cat /root/metrics/perf-bigann100m-anchor.txt 2>/dev/null || echo "No anchor perf data found"

plot-perf-counters:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make plot-perf-counters INPUT=results.json [OUTPUT=perf_comparison.png] [PERF_ONLY=1]"; \
		exit 1; \
	fi
	poetry run python plot_perf_counters.py \
		--input $(INPUT) \
		$(if $(PERF_ONLY),--perf-only,) \
		--output $(or $(OUTPUT),perf_comparison.png)

################################
# End Anchor Construction Perf Analysis
################################


################################
# Pareto Frontier Parameter Sweeps
# Sweep baseline and anchor parameters to build a Pareto frontier
# of construction time / distance computations vs recall.
# Run with: make pareto-sweep-sift / make pareto-sweep-yandex-deep
#           make pareto-sweep-all  (both datasets)
#           make pareto-plot INPUT=<json> OUTPUT=<png>
################################

pareto-sweep-sift:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 8 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--sweep \
		--sweep-anchor-probes 5 10 20 50 \
		--sweep-bulk-ef 50 80 100 \
		--sweep-baseline-ef 50 80 100 \
		--sweep-baseline-init 50 100 \
		--output /root/data/anchor-construction/sift-pareto-sweep.json

pareto-sweep-yandex-deep:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/yandex-deep-100m/yandex_100m.train.npy \
		--queries /root/data/yandex-deep-100m/yandex_100m.test.npy \
		--gtruth /root/data/yandex-deep-100m/yandex_100m.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--sweep \
		--sweep-anchor-probes 10 \
		--sweep-bulk-ef 80 \
		--sweep-baseline-ef 100 \
		--sweep-baseline-init 100 \
		--output /root/data/anchor-construction/yandex-deep-pareto-sweep.json


pareto-sweep-bigann-100m:
	poetry run python evaluate_two_pass.py \
		--dataset /root/data/bigann-100m-euclidean/bigann_100m.train.npy \
		--queries /root/data/bigann-100m-euclidean/bigann_100m.test.npy \
		--gtruth /root/data/bigann-100m-euclidean/bigann_100m.gtruth.npy \
		--strategies baseline anchor \
		--num-threads 32 \
		--M-baseline 16 \
		--anchor-fraction 0.01 \
		--anchor-M 16 \
		--anchor-ef-construction 500 \
		--sweep \
		--sweep-anchor-probes 10 \
		--sweep-bulk-ef 80 \
		--sweep-baseline-ef 100 \
		--sweep-baseline-init 100 \
		--output /root/data/anchor-construction/bigann-100m-pareto-sweep.json

pareto-sweep-all: pareto-sweep-sift pareto-sweep-yandex-deep

pareto-plot:
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make pareto-plot INPUT=sweep_results.json [OUTPUT=pareto.png]"; \
		exit 1; \
	fi
	poetry run python plot_pareto.py \
		--input $(INPUT) \
		--output $(or $(OUTPUT),pareto.png)

################################
# End Pareto Frontier Parameter Sweeps
################################

# Cachegrind analysis helpers
cachegrind-annotate-baseline:
	cg_annotate cachegrind.baseline.out

cachegrind-annotate-two-pass:
	cg_annotate cachegrind.two_pass.out

cachegrind-diff-two-pass:
	cg_diff cachegrind.baseline.out cachegrind.two_pass.out

sift-bench-hnsw: 
	poetry run python run-benchmark.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1


glove25-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename glove25.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove25-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove50-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-50-angular/glove-50-angular.train.npy \
		--queries /root/data/glove-50-angular/glove-50-angular.test.npy \
		--gtruth /root/data/glove-50-angular/glove-50-angular.gtruth.npy \
		--dataset-name glove-50-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename glove50.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove50-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-50-angular/glove-50-angular.train.npy \
		--queries /root/data/glove-50-angular/glove-50-angular.test.npy \
		--gtruth /root/data/glove-50-angular/glove-50-angular.gtruth.npy \
		--dataset-name glove-50-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove100-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename glove100.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove100-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove200-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-200-angular/glove-200-angular.train.npy \
		--queries /root/data/glove-200-angular/glove-200-angular.test.npy \
		--gtruth /root/data/glove-200-angular/glove-200-angular.gtruth.npy \
		--dataset-name glove-200-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename glove200.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove200-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/glove-200-angular/glove-200-angular.train.npy \
		--queries /root/data/glove-200-angular/glove-200-angular.test.npy \
		--gtruth /root/data/glove-200-angular/glove-200-angular.gtruth.npy \
		--dataset-name glove-200-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular


nytimes-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type flatnav \
		--use-hnsw-base-layer \
        --hnsw-base-layer-filename nytimes.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular


nytimes-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

################################
# End ANN Benchmarks
################################

setup: install-flatnav install-hnswlib

install-hnswlib:
	if [ ! -d "hnswlib-original" ]; then \
		git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git; \
	fi
	cd hnswlib-original/python_bindings && \
	poetry install --no-root && \
	poetry run python setup.py bdist_wheel 
	poetry add hnswlib-original/python_bindings/dist/*.whl

cleanup:
	rm -rf hnswlib-original 

# If passed an invalid argument, print help message
%:
	@echo "Invalid argument: $@"
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  setup: install all dependencies including flatnav"
	@echo "  install-flatnav: install flatnav"
	@echo "  install-hnswlib: install hnswlib"
	@echo "  cleanup: remove hnswlib-original"
	@echo "  generate-wheel: generate wheel for flatnav"
	@echo "  yandex-deep-bench: run yandex-deep benchmark"
	@echo "  sift-bench: run sift benchmark"
	
