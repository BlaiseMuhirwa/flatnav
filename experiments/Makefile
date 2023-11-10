
# Target to install poetry dependencies
install-dependencies:
	poetry install

# Target to run mnist benchmark
mnist: install-dependencies
	poetry run python evaluate_faiss.py \
		--datasets mnist-784-euclidean \
		--ef_cons 32 64 128 \
		--ef_search 32 64 128 256 512 \
		--num_node_links 8 16 32 64 128 \
		--pq_m 8 16

# Target to run sift benchmark
sift: install-dependencies
	poetry run python evaluate_faiss.py \
		--datasets sift-128-euclidean \
		--ef_cons 32 64 128 \
		--ef_search 32 64 128 256 512 \
		--num_node_links 8 16 32 64 128 \
		--pq_m 8 16 32 64

# Target to run gist benchmark
gist: install-dependencies
	poetry run python evaluate_faiss.py \
		--datasets gist-960-euclidean \
		--ef_cons 32 64 128 \
		--ef_search 32 64 128 256 512 \
		--num_node_links 8 16 32 64 128 \
		--pq_m 8 16 32 64

# Target for the glove-25 benchmark 
glove-25: install-dependencies
	poetry run python evaluate_faiss.py \
		--datasets glove-25-angular \
		--ef_cons 32 64 128 \
		--ef_search 32 64 128 256 512 \
		--num_node_links 8 16 32 64 128 \
		--pq_m 1 5

# Target for the glove-100 benchmark 
glove-100: install-dependencies
	poetry run python evaluate_faiss.py \
		--datasets glove-100-angular \
		--ef_cons 32 64 128 \
		--ef_search 32 64 128 256 512 \
		--num_node_links 8 16 32 64 128 \
		--pq_m 5 10 20 

# Target for the glove-200 benchmark 
glove-200: install-dependencies
	poetry run python evaluate_faiss.py \
		--datasets glove-200-angular \
		--ef_cons 32 64 128 \
		--ef_search 32 64 128 256 512 \
		--num_node_links 8 16 32 64 128 \
		--pq_m 5 10 20 40 



