import json
import logging
import time
import os
import nmslib
import hnswlib
import flatnav
import multiprocessing
from run_benchmark import train_index, compute_metrics
import argparse
import numpy as np

logger = logging.getLogger(__name__)

ROOT_DATA_PATH = "/root/data/"
REPRODUCIBILITY_EXPERIMENTS_PATH = os.path.join(
    ROOT_DATA_PATH, "reproducibility_experiments"
)
os.makedirs(REPRODUCIBILITY_EXPERIMENTS_PATH, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmarking HNSW and SWGraph")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10000000,
        help="Size of the dataset to index with.",
    )
    parser.add_argument(
        "--n-queries", type=int, default=10000, help="Number of queries to run."
    )
    parser.add_argument("--M", type=int, default=6, help="Maximum number of edges.")
    parser.add_argument(
        "--ef-construction",
        type=int,
        default=100,
        help="Number of neighbors to search.",
    )
    parser.add_argument(
        "--ef-search", type=int, default=100, help="Number of neighbors to search."
    )
    parser.add_argument(
        "--n-build-threads", type=int, default=1, help="Number of threads to use"
    )
    parser.add_argument(
        "--n-search-threads", type=int, default=1, help="Number of threads to use"
    )
    return parser.parse_args()


def generate_hypercube_dataset(size: int, d: int, save_path: str):
    # Check if the dataset already exists
    if os.path.exists(save_path):
        data = np.load(save_path)
    else:
        # Generate random data in the range [0, 1]
        data = np.random.uniform(0, 1, (size, d))
        np.save(save_path, data)

    # Select 1000 random query points from the data
    queries = data[np.random.choice(data.shape[0], 1000, replace=False)]
    np.save(save_path.replace(".npy", "_queries.npy"), queries)

    # Compute ground truth for the queries using optimized L2 distance calculation
    logger.info("Computing ground truth")
    
    # Step 1: Compute the squared norms of data and queries
    data_squared = np.sum(data ** 2, axis=1)
    queries_squared = np.sum(queries ** 2, axis=1)

    # Step 2: Compute the distance matrix using the squared Euclidean distance formula
    inner_product = np.dot(data, queries.T)
    distance_matrix = data_squared[:, None] + queries_squared[None, :] - 2 * inner_product

    # Step 3: Ensure no negative values in distance matrix before applying sqrt
    distances = np.sqrt(np.maximum(0, distance_matrix))

    # Get the top 10 nearest neighbors for each query
    k = 10
    top_k_indices = np.argpartition(distances, k, axis=0)[:k]  # Indices of the top 10 neighbors (unsorted)

    # Sort the top 10 indices by actual distance
    top_k_sorted = np.argsort(distances[top_k_indices, np.arange(distances.shape[1])], axis=0)
    top_k_sorted_indices = top_k_indices[top_k_sorted, np.arange(distances.shape[1])]

    # Save the ground truth nearest neighbors
    np.save(save_path.replace(".npy", "_ground_truth.npy"), top_k_sorted_indices)
    logger.info("Ground truth computed and saved")

def train_hnsw_index(
    train_data: np.ndarray, M: int, ef_construction: int, num_build_threads: int
) -> hnswlib.Index:
    logger.info("Training HNSW index")
    start = time.monotonic()
    index = train_index(
        train_dataset=train_data,
        dataset_size=train_data.shape[0],
        index_type="hnsw",
        distance_type="l2",
        max_edges_per_node=M,
        dim=train_data.shape[1],
        ef_construction=ef_construction,
        num_build_threads=num_build_threads,
    )
    end = time.monotonic()
    logger.info(f"Training HNSW index took {end - start:.4f} seconds")

    return index


def train_flatnav_index(
    train_data: np.ndarray, M: int, ef_construction: int, num_build_threads: int
) -> flatnav.index.IndexL2Float:
    logger.info("Training FlatNAV index")
    start = time.monotonic()
    index = train_index(
        train_dataset=train_data,
        dataset_size=train_data.shape[0],
        index_type="flatnav",
        distance_type="l2",
        max_edges_per_node=M,
        dim=train_data.shape[1],
        ef_construction=ef_construction,
        num_build_threads=num_build_threads,
    )
    end = time.monotonic()
    logger.info(f"Training FlatNAV index took {end - start:.4f} seconds")

    return index


def train_swgraph_index(
    train_data: np.ndarray, M: int, ef_construction: int, num_build_threads: int
):
    index = nmslib.init(
        method="sw-graph", space="l2", data_type=nmslib.DataType.DENSE_VECTOR
    )
    logger.info("Training SWGraph index")
    start = time.monotonic()
    index.addDataPointBatch(train_data)
    index_time_params = {'indexThreadQty': num_build_threads, 'efConstruction': ef_construction}
    index.createIndex(index_time_params)
    
    end = time.monotonic()

    logger.info(f"Training SWGraph index took {end - start:.4f} seconds")


def query_swgraph_index(
    index, queries: np.ndarray, ground_truth: int, ef_search: int, k: int
):
    logger.info("Querying SWGraph index")
    start = time.monotonic()
    query_time_params = {"efSearch": ef_search}
    index.setQueryTimeParams(query_time_params)
    results = index.knnQueryBatch(queries, k=k, num_threads=1)
    end = time.monotonic()
    logger.info(f"Querying SWGraph index took {end - start:.4f} seconds")


def main():
    args = parse_args()
    dimensions = [4, 8]
    for d in dimensions:
        full_path = os.path.join(
            REPRODUCIBILITY_EXPERIMENTS_PATH, f"iid_uniform_{d}.npy"
        )
        if not os.path.exists(full_path):
            raise ValueError(f"Dataset {full_path} does not exist")

        data = np.load(full_path)
        global_thread_count = args.n_build_threads
        per_index_thread_count = global_thread_count // 3

        print(f"Building swgraph index for {d} dimensions")
        swgraph_index = train_swgraph_index(
            train_data=data,
            M=args.M,
            ef_construction=args.ef_construction,
            num_build_threads=per_index_thread_count,
        )

        print(f"Building hnsw and flatnav index for {d} dimensions")
        hnsw_index = train_hnsw_index(
            train_data=data,
            M=args.M,
            ef_construction=args.ef_construction,
            num_build_threads=per_index_thread_count,
        )

        print(f"Building flatnav index for {d} dimensions")
        flatnav_index = train_flatnav_index(
            train_data=data,
            M=args.M,
            ef_construction=args.ef_construction,
            num_build_threads=per_index_thread_count,
        )


        # Select a sample of 10k queries randomly from teh dataset
        queries = data[np.random.choice(data.shape[0], args.n_queries, replace=False)]
        ground_truth = np.load(full_path.replace(".npy", "_ground_truth.npy"))

        ef_search_values = [100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000]

        metrics_dict = {}

        for ef_search in ef_search_values:
            logger.info("Searching with ef_search=%d", ef_search)
            print("Searching with ef_search=%d", ef_search)
            # Compute recall, distance computations and latency
            hnsw_metrics: dict[str, float] = compute_metrics(
                requested_metrics=["recall", "distance_computations", "latency_p50"],
                index=hnsw_index,
                queries=queries,
                ef_search=ef_search,
                ground_truth=ground_truth,
                k=10,
            )
            flatnav_metrics: dict[str, float] = compute_metrics(
                requested_metrics=["recall", "distance_computations", "latency_p50"],
                index=flatnav_index,
                queries=queries,
                ef_search=ef_search,
                ground_truth=ground_truth,
                k=10,
            )

            # Add metrics to the dictionary 
            if "flatnav" not in metrics_dict:
                metrics_dict["flatnav"] = {}
            if "hnsw" not in metrics_dict:
                metrics_dict["hnsw"] = {}
            if "swgraph" not in metrics_dict:
                metrics_dict["swgraph"] = {}

            metrics_dict["flatnav"].update(flatnav_metrics)
            metrics_dict["hnsw"].update(hnsw_metrics)

        # Serialize the metrics dictionary to disk as JSON 
        metrics_path = os.path.join(
            REPRODUCIBILITY_EXPERIMENTS_PATH, f"metrics_{d}.json"
        )
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f)




if __name__ == "__main__":
    logger.info("Starting experiments")
    main()