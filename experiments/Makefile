

# Run the yandex-deep benchmark and log everything to a logs.txt file 
yandex-deep-bench: 
	poetry run python run-big-bench.py \
		--dataset /media/scratch/yandex-deep/learn.350.fbin \
		--queries /media/scratch/yandex-deep/queries.fbin \
		--gtruth /media/scratch/yandex-deep/ground_truth.10K.bin \
		--metric l2 > logs.txt 2>&1

glove-bench:
	poetry run python run-big-bench.py \
		--dataset /root/data/glove-25-angular/glove-25-angular.train.npy \
		--queries /root/data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth /root/data/glove-25-angular/glove-25-angular.gtruth.npy \
		--use-hnsw-base-layer \
		--hnsw-base-layer-filename glove.mtx \
		--num-node-links 32 \
		--ef-construction 100 200 \
		--ef-search 100 200 300 \
		--metric angular

# Other available options include: --index-type (hnsw, flatnav)
sift-bench: 
	poetry run python run-big-bench.py \
		--dataset /root/data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries /root/data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth /root/data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--num-node-links 32 \
		--ef-construction 100 200 \
		--ef-search 100 200 300 \
		--metric l2 \
		--num-build-threads 16 \
		--num-search-threads 16 

hubness-test:
	poetry run python hubness.py \
		--datasets normal-10-angular normal-256-angular normal-1024-angular normal-10-euclidean normal-256-euclidean normal-1024-euclidean \
		--k 100 \
		--metrics angular angular angular l2 l2 l2 \
		--ef-construction 100 \
		--ef-search 200 \
		--num-node-links 32

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
	
