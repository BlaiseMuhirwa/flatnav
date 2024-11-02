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


def generate_hypercube_dataset(size: int, d: int, save_path: str, batch_size: int = 500):
    # Check if the dataset already exists
    if os.path.exists(save_path):
        data = np.load(save_path)
    else:
        # Generate random data in the range [0, 1]
        data = np.random.uniform(0, 1, (size, d))
        np.save(save_path, data)

    # Select 5000 random query points from the data
    queries = data[np.random.choice(data.shape[0], 10000, replace=False)]
    np.save(save_path.replace(".npy", f"_queries_{d}.npy"), queries)

    # Compute squared norms of data points for distance calculation
    print("Computing ground truth")
    print("Step 1: Compute the squared norms of data")
    data_squared = np.sum(data**2, axis=1)

    k = 10  # Top k nearest neighbors
    num_batches = int(np.ceil(queries.shape[0] / batch_size))
    top_k_sorted_indices = np.empty((k, queries.shape[0]), dtype=int)

    # Process each batch of queries
    for i in range(num_batches):
        print(f"Processing batch {i + 1} of {num_batches}")
        
        # Define the batch range
        start = i * batch_size
        end = min(start + batch_size, queries.shape[0])
        batch_queries = queries[start:end]
        batch_queries_squared = np.sum(batch_queries**2, axis=1)

        # Compute the distance matrix for the current batch
        inner_product_batch = np.dot(data, batch_queries.T)
        distance_matrix_batch = (
            data_squared[:, None] + batch_queries_squared[None, :] - 2 * inner_product_batch
        )
        distances_batch = np.sqrt(np.maximum(0, distance_matrix_batch))

        # Get the top k indices for each query in the batch
        top_k_indices_batch = np.argpartition(distances_batch, k, axis=0)[:k]

        # Sort the top k indices by actual distance
        top_k_sorted_batch = np.argsort(
            distances_batch[top_k_indices_batch, np.arange(distances_batch.shape[1])], axis=0
        )
        top_k_sorted_indices[:, start:end] = top_k_indices_batch[top_k_sorted_batch, np.arange(distances_batch.shape[1])]

    # Save the ground truth nearest neighbors
    np.save(save_path.replace(".npy", f"_ground_truth_{d}.npy"), top_k_sorted_indices)
    print("Ground truth computed and saved")



def train_hnsw_index(
    train_data: np.ndarray, M: int, ef_construction: int, num_build_threads: int
) -> hnswlib.Index:

    index_save_path = os.path.join(
        REPRODUCIBILITY_EXPERIMENTS_PATH, f"hnsw_index_{train_data.shape[1]}.bin"
    )

    if os.path.exists(index_save_path):
        print(f"Loading HNSW index from {index_save_path}")
        index = hnswlib.Index(space="l2", dim=train_data.shape[1])
        index.load_index(index_save_path)
        return index

    print("Training HNSW index")
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
    print(f"Training HNSW index took {end - start:.4f} seconds")

    # Save index to disk
    index.save_index(index_save_path)

    return index


def train_flatnav_index(
    train_data: np.ndarray, M: int, ef_construction: int, num_build_threads: int
) -> flatnav.index.IndexL2Float:

    index_save_path = os.path.join(
        REPRODUCIBILITY_EXPERIMENTS_PATH, f"flatnav_index_{train_data.shape[1]}.bin"
    )

    if os.path.exists(index_save_path):
        print(f"Loading FlatNAV index from {index_save_path}")
        index = flatnav.index.IndexL2Float.load_index(filename=index_save_path)
        return index

    print("Training FlatNAV index")
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
    print(f"Training FlatNAV index took {end - start:.4f} seconds")

    # Save index
    index.save(filename=index_save_path)

    return index


def train_swgraph_index(
    train_data: np.ndarray, M: int, ef_construction: int, num_build_threads: int
):
    data_dim = train_data.shape[1]
    index_save_path = os.path.join(
        REPRODUCIBILITY_EXPERIMENTS_PATH, f"swgraph_index_{data_dim}.bin"
    )
    if os.path.exists(index_save_path):
        print(f"Loading SWGraph index from {index_save_path}")
        index = nmslib.init(
            method="sw-graph", space="l2", data_type=nmslib.DataType.DENSE_VECTOR
        )
        index.loadIndex(index_save_path, load_data=True)
        return index
    index = nmslib.init(
        method="sw-graph", space="l2", data_type=nmslib.DataType.DENSE_VECTOR
    )
    print("Training SWGraph index")
    start = time.monotonic()
    index.addDataPointBatch(train_data)
    index_time_params = {
        "NN": 6,
        "indexThreadQty": num_build_threads,
        "efConstruction": ef_construction,
    }
    index.createIndex(index_time_params)

    end = time.monotonic()

    print(f"Training SWGraph index took {end - start:.4f} seconds")
    index.saveIndex(index_save_path, save_data=True)
    return index


def query_swgraph_index(
    index, queries: np.ndarray, ground_truth: np.ndarray, ef_search: int, k: int
) -> dict[str, float]:

    def compute_recall(query_results, ground_truth):
        recall = 0
        assert query_results.shape == ground_truth.shape
        for query_result, gt in zip(query_results, ground_truth):
            recall += len(set(query_result).intersection(gt)) / len(gt)
        return recall / len(ground_truth)

    print("Querying SWGraph index")
    start = time.monotonic()
    query_time_params = {"efSearch": ef_search}
    index.setQueryTimeParams(query_time_params)

    latencies = []
    query_results = []
    for query in queries:
        start = time.monotonic()
        ids, _ = index.knnQuery(query, k=k)
        end = time.monotonic()
        query_results.append(ids)
        latencies.append(end - start)

    query_results = np.array(query_results)
    recall = compute_recall(query_results, ground_truth)

    metrics = {
        "recall": recall,
        "latency_p50": np.percentile(latencies, 50) * 1000,
        "latency_avg": np.mean(latencies) * 1000,
    }
    return metrics


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
        # flatnav_index = train_flatnav_index(
        #     train_data=data,
        #     M=args.M,
        #     ef_construction=args.ef_construction,
        #     num_build_threads=per_index_thread_count,
        # )

        # Select a sample of 10k queries randomly from teh dataset
        queries = np.load(full_path.replace(".npy", f"_queries_{d}.npy"))
        # Cast queries to float32
        queries = queries.astype(np.float32)

        ground_truth = np.load(full_path.replace(".npy", f"_ground_truth_{d}.npy"))
        ground_truth = ground_truth.astype(np.int32)

        # print type of queries and ground truth
        print("Queries shape: ", queries.shape)
        print(f"Queries type: {queries.dtype}")

        print("Ground truth shape: ", ground_truth.shape)
        print(f"Ground truth type: {ground_truth.dtype}")


        # Reshape the ground truth to match the queries
        ground_truth = ground_truth.T

        ef_search_values = [20, 50, 80, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

        metrics_dict = {}
        metrics_path = os.path.join(
            REPRODUCIBILITY_EXPERIMENTS_PATH, f"metrics_{d}.json"
        )

        for ef_search in ef_search_values:
            print(f"Searching with ef_search={ef_search}")
            # Compute recall, distance computations and latency
            hnsw_metrics: dict[str, float] = compute_metrics(
                requested_metrics=["recall", "latency_avg", "latency_p50"],
                index=hnsw_index,
                queries=queries,
                ef_search=ef_search,
                ground_truth=ground_truth,
                k=10,
            )
            # flatnav_metrics: dict[str, float] = compute_metrics(
            #     requested_metrics=["recall", "latency_avg", "latency_p50"],
            #     index=flatnav_index,
            #     queries=queries,
            #     ef_search=ef_search,
            #     ground_truth=ground_truth,
            #     k=10,
            # )

            # Query sw-graph
            sw_graph_metrics = query_swgraph_index(
                index=swgraph_index,
                queries=queries,
                ground_truth=ground_truth,
                ef_search=args.ef_search,
                k=10,
            )

            # Add metrics to the dictionary
            if "flatnav" not in metrics_dict:
                metrics_dict["flatnav"] = []
            if "hnsw" not in metrics_dict:
                metrics_dict["hnsw"] = []
            if "swgraph" not in metrics_dict:
                metrics_dict["swgraph"] = []

            # metrics_dict["flatnav"].append(flatnav_metrics)
            metrics_dict["hnsw"].append(hnsw_metrics)
            metrics_dict["swgraph"].append(sw_graph_metrics)

        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f)

if __name__ == "__main__":
    print("Starting experiments")
    main()