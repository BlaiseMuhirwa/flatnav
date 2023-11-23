import time
from typing import Union
import json
import numpy as np
from typing import Optional, Tuple, List
import numpy as np
from dvclive import Live
import os
import logging
import platform, socket, psutil
import argparse
import flatnav


ENVIRONMENT_INFO = {
    "load_before_experiment": os.getloadavg()[2],
    "platform": platform.platform(),
    "platform_version": platform.version(),
    "platform_release": platform.release(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "hostname": socket.gethostname(),
    "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
    "num_cores": psutil.cpu_count(logical=True),
}


def load_benchmark_dataset(
    train_dataset_path: str, queries_path: str, gtruth_path: str
) -> Tuple[np.ndarray]:
    def verify_paths_exist(paths: List[str]) -> None:
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Invalid file path: {path}")

    verify_paths_exist([train_dataset_path, queries_path, gtruth_path])

    train_dataset = np.fromfile(
        train_dataset_path,
        dtype=np.float32 if train_dataset_path.endswith("fbin") else np.uint8,
    )
    queries_dataset = np.fromfile(
        queries_path, dtype=np.float32 if queries_path.endswith("fbin") else np.uint8
    )
    gtruth_dataset = np.fromfile(
        gtruth_path,
        dtype=np.float32 if gtruth_path.endswith("fbin") else np.uint8,
    )

    return train_dataset, queries_dataset, gtruth_dataset


def compute_metrics(
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k=100,
) -> Tuple[float, float]:
    """
    Compute recall and QPS for given queries, ground truth, and a FlaNav index.

    Args:
        - index: The Faiss index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.

    Returns:
        Mean recall over all queries.
        QPS over all queries
    """
    start = time.time()
    _, top_k_indices = index.search(queries=queries, ef_search=ef_search, K=k)
    end = time.time()

    querying_time = end - start
    qps = 1 / querying_time

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)

    return recall, qps


def train_flatnav_index(
    train_dataset: np.ndarray,
    distance_type: str,
    dim: int,
    dataset_size: int,
    max_edges_per_node: int,
    ef_construction: int,
) -> Union[flatnav.index.L2Index, flatnav.index.IPIndex]:
    index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
    )

    # Train the index.
    start = time.time()
    index.add(data=train_dataset, ef_construction=ef_construction)
    end = time.time()

    logging.info(f"Indexing time = {end - start} seconds")

    return index


def main(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    gtruth: np.ndarray,
    ef_cons_params: List[int],
    ef_search_params: List[int],
    num_node_links: List[int],
    distance_type: str,
):
    dataset_size = train_dataset.shape[0]
    dim = train_dataset.shape[1]

    for node_links in num_node_links:
        for ef_cons in ef_cons_params:
            for ef_search in ef_search_params:
                index = train_flatnav_index(
                    train_dataset=train_dataset,
                    max_edges_per_node=node_links,
                    ef_construction=ef_cons,
                    dataset_size=dataset_size,
                    dim=dim,
                    distance_type=distance_type,
                )

                recall, qps = compute_metrics(
                    index=index,
                    queries=queries,
                    ground_truth=gtruth,
                    ef_search=ef_search,
                )

                logging.info(
                    f"Recall@100: {recall}, QPS={qps}, node_links={node_links},"
                    f" ef_cons={ef_cons}, ef_search={ef_search}"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description="Benchmark Flatnav on Big ANN datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a single ANNS benchmark dataset to run on.",
    )
    parser.add_argument(
        "--queries", required=True, help="Path to a singe queries file."
    )
    parser.add_argument(
        "--gtruth",
        required=True,
        help="Path to a single ground truth file to evaluate on.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        default="l2",
        help="Distance tye. Options include `l2` and `angular`.",
    )
    parser.add_argument(
        "--log_metrics", required=False, default=False, help="Log metrics to DVC."
    )

    args = parser.parse_args()

    ef_construction_params = [32, 64, 128]
    ef_search_params = [32, 64, 128, 256]
    num_node_links = [8, 16, 32, 64]

    train_data, queries, ground_truth = load_benchmark_dataset(
        train_dataset_path=args.dataset,
        queries_path=args.queries,
        gtruth_path=args.gtruth,
    )

    main(
        train_dataset=train_data,
        queries=queries,
        gtruth=ground_truth,
        ef_cons_params=ef_construction_params,
        ef_search_params=ef_search_params,
        num_node_links=num_node_links,
        distance_type=args.metric.lower(),
    )
