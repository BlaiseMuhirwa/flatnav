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
REPRODUCIBILITY_EXPERIMENTS_PATH = os.path.join(ROOT_DATA_PATH, "reproducibility_experiments")
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
    parser.add_argument("--M", type=int, default=32, help="Maximum number of edges.")
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
    data = np.random.uniform(0, 1, (size, d))
    np.save(save_path, data)


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
    index.createIndex(
        {"efConstruction": ef_construction, "indexThreadQty": num_build_threads}
    )
    end = time.monotonic()

    logger.info(f"Training SWGraph index took {end - start:.4f} seconds")


def main():
    args = parse_args()
    dimensions = [4, 8]
    for d in dimensions:
        full_path = os.path.join(REPRODUCIBILITY_EXPERIMENTS_PATH, f"iid_uniform_{d}.npy")
        if not os.path.exists(full_path):
            generate_hypercube_dataset(args.dataset_size, d, full_path)

        data = np.load(full_path)
        global_thread_count = args.n_build_threads
        per_index_thread_count = global_thread_count // 3

        # Use multiprocessing to train a HNSW, FlatNav and SWGraph index
        hnsw_process = multiprocessing.Process(
            target=train_hnsw_index,
            args=(data, args.M, args.ef_construction, per_index_thread_count),
        )
        flatnav_process = multiprocessing.Process(
            target=train_flatnav_index,
            args=(data, args.M, args.ef_construction, per_index_thread_count),
        )

        swgraph_process = multiprocessing.Process(
            target=train_swgraph_index,
            args=(data, args.M, args.ef_construction, per_index_thread_count),
        )

        hnsw_process.start()
        flatnav_process.start()
        swgraph_process.start()

        hnsw_process.join()
        flatnav_process.join()
        swgraph_process.join()


if __name__=="__main__":
    main()