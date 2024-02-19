

# Run the yandex-deep benchmark and log everything to a logs.txt file 
yandex-deep-bench: 
	poetry run python run-big-bench.py \
		--dataset /media/scratch/yandex-deep/learn.350.fbin \
		--queries /media/scratch/yandex-deep/queries.fbin \
		--gtruth /media/scratch/yandex-deep/ground_truth.10K.bin \
		--metric l2 > logs.txt 2>&1

# Other available options include: --index-type (hnsw, flatnav)
# We use dataset-name for the name of the plots.

fashion-mnist-bench-flatnav: 
	poetry run python run-big-bench.py \
		--dataset-name fashion-mnist-784 \
		--dataset /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.train.npy \
		--queries /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.test.npy \
		--gtruth /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename fashion-mnist.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


fashion-mnist-bench-hnsw: 
	poetry run python run-big-bench.py \
		--dataset-name fashion-mnist-784 \
		--dataset /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.train.npy \
		--queries /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.test.npy \
		--gtruth /root/data/fashion-mnist-784-euclidean/fashion-mnist-784-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

gist-bench-flatnav: 
	poetry run python run-big-bench.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename gist.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 


gist-bench-hnsw: 
	poetry run python run-big-bench.py \
		--dataset-name gist \
		--dataset /root/data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries /root/data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth /root/data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 
sift-bench-flatnav: 
	poetry run python run-big-bench.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type flatnav \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename sift.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1 

sift-bench-hnsw: 
	poetry run python run-big-bench.py \
		--dataset-name sift \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 1

glove100-bench-flatnav:
	poetry run python run-big-bench.py \
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove100-bench-hnsw:
	poetry run python run-big-bench.py \
		--dataset /root/data/glove-100-angular/glove-100-angular.train.npy \
		--queries /root/data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth /root/data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove-100-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular


glove25-bench-flatnav:
	poetry run python run-big-bench.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

glove25-bench-hnsw:
	poetry run python run-big-bench.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove-25-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 3000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

nytimes-bench-flatnav:
	poetry run python run-big-bench.py \
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type flatnav \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 2500 5000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

nytimes-bench-hnsw:
	poetry run python run-big-bench.py \
		--dataset /root/data/nytimes-256-angular/nytimes-256-angular.train.npy \
		--queries /root/data/nytimes-256-angular/nytimes-256-angular.test.npy \
		--gtruth /root/data/nytimes-256-angular/nytimes-256-angular.gtruth.npy \
		--dataset-name nytimes-256-angular \
		--index-type hnsw \
		--num-node-links 32 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 500 1000 2500 5000 \
		--num-build-threads 16 \
		--num-search-threads 1 \
		--metric angular

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
	
