

sift-pruning-job:
	@echo "Running SIFT pruning job..."
	python runner.py \
		--dataset ../data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries ../data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth ../data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--dataset-name sift \
		--metric l2 \
		--num-node-links 16 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 400 500 1000 2000 \
		--k-neighbors 100 \
		--num-build-threads 16

glove25-pruning-job:
	@echo "Running GloVe 25 pruning job..."
	python runner.py \
		--dataset ../data/glove-25-angular/glove-25-angular.train.npy \
		--queries ../data/glove-25-angular/glove-25-angular.test.npy \
		--gtruth ../data/glove-25-angular/glove-25-angular.gtruth.npy \
		--dataset-name glove25 \
		--metric l2 \
		--num-node-links 16 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 400 500 1000 2000 \
		--k-neighbors 100 \
		--num-build-threads 16

glove100-pruning-job:
	@echo "Running GloVe 100 pruning job..."
	python runner.py \
		--dataset ../data/glove-100-angular/glove-100-angular.train.npy \
		--queries ../data/glove-100-angular/glove-100-angular.test.npy \
		--gtruth ../data/glove-100-angular/glove-100-angular.gtruth.npy \
		--dataset-name glove100 \
		--metric l2 \
		--num-node-links 16 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 400 500 1000 2000 \
		--k-neighbors 100 \
		--num-build-threads 16


glove200-pruning-job:
	@echo "Running GloVe 200 pruning job..."
	python runner.py \
		--dataset ../data/glove-200-angular/glove-200-angular.train.npy \
		--queries ../data/glove-200-angular/glove-200-angular.test.npy \
		--gtruth ../data/glove-200-angular/glove-200-angular.gtruth.npy \
		--dataset-name glove200 \
		--metric l2 \
		--num-node-links 16 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 400 500 1000 2000 \
		--k-neighbors 100 \
		--num-build-threads 16

gist960-pruning-job:
	@echo "Running GIST 960 pruning job..."
	python runner.py \
		--dataset ../data/gist-960-euclidean/gist-960-euclidean.train.npy \
		--queries ../data/gist-960-euclidean/gist-960-euclidean.test.npy \
		--gtruth ../data/gist-960-euclidean/gist-960-euclidean.gtruth.npy \
		--dataset-name gist \
		--metric l2 \
		--num-node-links 16 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 400 500 1000 2000 \
		--k-neighbors 100 \
		--num-build-threads 16

mnist-pruning-job:
	@echo "Running MNIST pruning job..."
	python runner.py \
		--dataset ../data/mnist-784-euclidean/mnist-784-euclidean.train.npy \
		--queries ../data/mnist-784-euclidean/mnist-784-euclidean.test.npy \
		--gtruth ../data/mnist-784-euclidean/mnist-784-euclidean.gtruth.npy \
		--dataset-name mnist \
		--metric l2 \
		--num-node-links 16 \
		--ef-construction 100 200 300 \
		--ef-search 100 200 300 400 500 1000 2000 \
		--k-neighbors 100 \
		--num-build-threads 16