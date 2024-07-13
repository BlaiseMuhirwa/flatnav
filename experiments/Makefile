# This file contains the benchmarking script configurations for various near neighbor datasets
# The main configurable parameter is the type of index: --index-type (hnsw, flatnav)
# We use dataset-name for the name of the plots.

################################
# Begin Billion-Scale Benchmarks
################################

# Billion-scale benchmark for the Microsoft SpaceV1B dataset
# Reference: https://github.com/microsoft/SPTAG/tree/main/datasets/SPACEV1B

space-v1b-bench-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name msft-spacev-1b \
		--dataset /root/data/SPACEV1B/collection.npy \
		--queries /root/data/SPACEV1B/queries.npy \
		--gtruth /root/data/SPACEV1B/ground_truth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 30 50 100 \
		--ef-search 100 200 300 500 1000 \
		--num-build-threads 32 \
		--num-search-threads 1 \
		--metric l2


space-v1b-bench-flatnav:
	poetry run python run-benchmark.py \
		--dataset-name msft-spacev-1b \
		--dataset /root/data/SPACEV1B/collection.npy \
		--queries /root/data/SPACEV1B/queries.npy \
		--gtruth /root/data/SPACEV1B/ground_truth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename space-v1b.mtx \
		--num-node-links 32 \
		--ef-construction 30 50 100 \
		--ef-search 100 200 300 500 1000 \
		--num-build-threads 45 \
		--num-search-threads 1 \
		--metric l2

################################
# End Billion-Scale Benchmarks
################################

################################
# Begin 100-Million Scale Benchmarks
# ################################
bigann-100m-hnsw: 
	poetry run python run-benchmark.py \
                --dataset-name bigann-100m \
                --dataset /root/data/bigann-100m-euclidean/bigann_train_100m.npy \
                --queries /root/data/bigann-100m-euclidean/bigann_queries.npy \
                --gtruth /root/data/bigann-100m-euclidean/bigann_ground_truth_100m.npy \
                --index-type flatnav \
                --num-node-links 32 \
                --ef-construction 50 100 200 \
                --ef-search 100 200 300 500 1000 3000 \
                --metric l2 \
                --num-build-threads 32 \
                --num-search-threads 1


bigann-100m-flatnav: 
	poetry run python run-benchmark.py \
                --dataset-name bigann-100m \
                --dataset /root/data/bigann-100m-euclidean/bigann_train_100m.npy \
                --queries /root/data/bigann-100m-euclidean/bigann_queries.npy \
                --gtruth /root/data/bigann-100m-euclidean/bigann_ground_truth_100m.npy \
                --index-type flatnav \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename bigann-100m.mtx \
                --num-node-links 32 \
                --ef-construction 50 100 200 \
                --ef-search 100 200 300 500 1000 3000 \
                --metric l2 \
                --num-build-threads 32 \
                --num-search-threads 1


yandex-deep-100m-hnsw:
	poetry run python run-benchmark.py \
                --dataset-name yandex-deep-100m \
                --dataset /root/data/yandex-deep-100m/yandex_train_100m.npy \
                --queries /root/data/yandex-deep-100m/yandex_queries.npy \
                --gtruth /root/data/yandex-deep-100m/yandex_ground_truth_100m.npy \
            	--index-type hnsw \
                --num-node-links 32 \
                --ef-construction 50 100 200 \
                --ef-search 100 200 300 500 1000 3000 \
                --metric l2 \
                --num-build-threads 32 \
                --num-search-threads 1


yandex-deep-100m-flatnav:
	poetry run python run-benchmark.py \
                --dataset-name yandex-deep-100m \
                --dataset /root/data/yandex-deep-100m/yandex_train_100m.npy \
                --queries /root/data/yandex-deep-100m/yandex_queries.npy \
                --gtruth /root/data/yandex-deep-100m/yandex_ground_truth_100m.npy \
            	--index-type flatnav \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename yandex-deep-100m.mtx \
                --num-node-links 32 \
                --ef-construction 50 100 200 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric l2 \
        	--num-build-threads 32 \
        	--num-search-threads 1


spacev-100m-flatnav:
	poetry run python run-benchmark.py \
                --dataset-name spacev-100m \
                --dataset /root/data/spacev-100m/spacev_train_100m.npy \
                --queries /root/data/spacev-100m/spacev_queries.npy \
                --gtruth /root/data/spacev-100m/spacev_ground_truth_100m.npy \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename spacev-100m.mtx \
                --index-type flatnav \
                --num-node-links 32 \
                --ef-construction 50 100 200 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric l2 \
        	--num-build-threads 16 \
        	--num-search-threads 1

spacev-100m-hnsw:
	poetry run python run-benchmark.py \
                --dataset-name spacev-100m \
                --dataset /root/data/spacev-100m/spacev_train_100m.npy \
                --queries /root/data/spacev-100m/spacev_queries.npy \
                --gtruth /root/data/spacev-100m/spacev_ground_truth_100m.npy \
            	--index-type hnsw \
                --num-node-links 32 \
                --ef-construction 50 100 200 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric l2 \
        	--num-build-threads 32 \
        	--num-search-threads 1


tti-100m-flatnav:
	poetry run python run-benchmark.py \
        	--dataset-name tti-100m \
        	--dataset /root/data/tti-100m/tti_train_100m.npy \
        	--queries /root/data/tti-100m/tti_queries.npy \
        	--gtruth /root/data/tti-100m/tti_ground_truth_100m.npy \
        	--index-type flatnav \
        	--use-hnsw-base-layer \
        	--hnsw-base-layer-filename tti-100m.mtx \
        	--num-node-links 32 \
        	--ef-construction 50 100 200 \
        	--ef-search 100 200 300 500 1000 3000 \
        	--metric angular \
        	--num-build-threads 32 \
        	--num-search-threads 1

tti-100m-hnsw:
	poetry run python run-benchmark.py \
		--dataset-name tti-100m \
                --dataset /root/data/tti-100m/tti_train_100m.npy \
                --queries /root/data/tti-100m/tti_queries.npy \
                --gtruth /root/data/tti-100m/tti_ground_truth_100m.npy \
                --index-type hnsw \
                --num-node-links 32 \
                --ef-construction 50 100 200 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric angular \
        	--num-build-threads 32 \
        	--num-search-threads 1

################################
# End 100-Million Scale Benchmarks
# # ################################

################################
# Begin 10-Million Scale Benchmarks
# # ################################
bigann-10m-flatnav: 
	poetry run python run-benchmark.py \
                --dataset-name bigann-10m \
                --dataset /root/data/bigann-10m-euclidean/bigann_train_10m.npy \
                --queries /root/data/bigann-10m-euclidean/bigann_queries.npy \
                --gtruth /root/data/bigann-10m-euclidean/bigann_ground_truth_10m.npy \
                --index-type flatnav \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename bigann-10m.mtx \
                --num-node-links 32 \
                --ef-construction 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
                --metric l2 \
                --num-build-threads 16 \
                --num-search-threads 1


bigann-10m-hnsw:
	poetry run python run-benchmark.py \
                --dataset-name bigann-10m \
                --dataset /root/data/bigann-10m-euclidean/bigann_train_10m.npy \
                --queries /root/data/bigann-10m-euclidean/bigann_queries.npy \
                --gtruth /root/data/bigann-10m-euclidean/bigann_ground_truth_10m.npy \
                --index-type hnsw \
                --num-node-links 32 \
                --ef-construction 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
                --metric l2 \
                --num-build-threads 16 \
                --num-search-threads 1


yandex-deep-10m-flatnav:
	poetry run python run-benchmark.py \
                --dataset-name yandex-deep-10m \
                --dataset /root/data/yandex-deep-10m/yandex_train_10m.npy \
                --queries /root/data/yandex-deep-10m/yandex_queries.npy \
                --gtruth /root/data/yandex-deep-10m/yandex_ground_truth_10m.npy \
            	--index-type flatnav \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename yandex-deep-10m.mtx \
                --num-node-links 32 \
                --ef-construction 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric l2 \
        	--num-build-threads 16 \
        	--num-search-threads 1


yandex-deep-10m-hnsw:
	poetry run python run-benchmark.py \
                --dataset-name yandex-deep-10m \
                --dataset /root/data/yandex-deep-10m/yandex_train_10m.npy \
                --queries /root/data/yandex-deep-10m/yandex_queries.npy \
                --gtruth /root/data/yandex-deep-10m/yandex_ground_truth_10m.npy \
            	--index-type hnsw \
                --num-node-links 32 \
                --ef-construction 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
                --metric l2 \
                --num-build-threads 16 \
                --num-search-threads 1


spacev-10m-flatnav:
	poetry run python run-benchmark.py \
                --dataset-name spacev-10m \
                --dataset /root/data/spacev-10m/spacev_train_10m.npy \
                --queries /root/data/spacev-10m/spacev_queries.npy \
                --gtruth /root/data/spacev-10m/spacev_ground_truth_10m.npy \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename spacev-10m.mtx \
                --index-type flatnav \
                --num-node-links 32 \
                --ef-construction 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric l2 \
        	--num-build-threads 38 \
        	--num-search-threads 1


spacev-10m-hnsw:
	poetry run python run-benchmark.py \
                --dataset-name spacev-10m \
                --dataset /root/data/spacev-10m/spacev_train_10m.npy \
                --queries /root/data/spacev-10m/spacev_queries.npy \
                --gtruth /root/data/spacev-10m/spacev_ground_truth_10m.npy \
            	--index-type hnsw \
                --num-node-links 32 \
                --ef-construction 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric l2 \
        	--num-build-threads 16 \
        	--num-search-threads 1


tti-10m-flatnav:
	poetry run python run-benchmark.py \
                --dataset-name tti-10m \
                --dataset /root/data/tti_10m/tti_train_10m.npy \
                --queries /root/data/tti_10m/tti_queries.npy \
                --gtruth /root/data/tti_10m/tti_ground_truth_10m.npy \
            	--index-type flatnav \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename tti-10m.mtx \
                --num-node-links 32 \
                --ef-construction 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
        	--metric angular \
        	--num-build-threads 32 \
        	--num-search-threads 1

tti-10m-hnsw:
	poetry run python run-benchmark.py \
                --dataset-name tti-10m \
                --dataset /root/data/tti_10m/tti_train_10m.npy \
                --queries /root/data/tti_10m/tti_queries.npy \
                --gtruth /root/data/tti_10m/tti_ground_truth_10m.npy \
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

deep-image-flatnav: 
	poetry run python run-benchmark.py \
		--dataset-name deep-image-96 \
		--dataset /root/data/deep-image-96-angular/deep-image-96-angular.train.npy \
		--queries /root/data/deep-image-96-angular/deep-image-96-angular.test.npy \
		--gtruth /root/data/deep-image-96-angular/deep-image-96-angular.gtruth.npy \
		--index-type flatnav \
                --use-hnsw-base-layer \
                --hnsw-base-layer-filename deep-image.mtx \
                --num-node-links 32 \
                --ef-construction 30 40 50 100 200 300 \
                --ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
                --num-search-threads 1

deep-image-hnsw: 
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
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename sift.mtx \
		--num-node-links 32 \
		--ef-construction 30 40 50 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


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

# Install all dependencies including flatnav
install-flatnav: generate-wheel 
	poetry add ../flatnav_python/dist/*.whl 
	rm poetry.lock && poetry install --no-root

# This will generate the wheel for flatnav and put it in
# ../flatnav_python/dist
generate-wheel:
	poetry remove flatnav &> /dev/null 
	cd .. && cd flatnav_python && ./install_flatnav.sh 

install-hnswlib:
	if [ ! -d "hnswlib-original" ]; then \
		git clone https://github.com/BlaiseMuhirwa/hnswlib-original.git; \
	fi
	cd hnswlib-original/python_bindings && \
	poetry install --no-root && \
	poetry run python setup.py bdist_wheel 
	poetry add hnswlib-original/python_bindings/dist/*.whl

s3-push:
	poetry run python push-snapshot-to-s3.py

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
	
