

sift-pruning-job:
	@echo "Running SIFT pruning job..."
	python runner.py \
		--dataset ../data/sift-128-euclidean/sift-128-euclidean.train.npy \
		--queries ../data/sift-128-euclidean/sift-128-euclidean.test.npy \
		--gtruth ../data/sift-128-euclidean/sift-128-euclidean.gtruth.npy \
		--dataset-name sift \
		--metric l2 \
		--num-node-links 16 32 \
		--ef-construction 100 200 300 400 500 \
		--ef-search 100 200 300 400 500 \
		--k-neighbors 100 \
		--num-build-threads 16