from typing import Tuple, Union
import numpy as np
import hnswlib
import time
import flatnav


def compute_metrics(
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex, hnswlib.Index],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k=100,
) -> Tuple[float, float]:
    """
    Compute recall and QPS for given queries, ground truth for the given index(FlatNav or HNSW).

    Args:
        - index: A FlatNav index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.

    Returns:
        Mean recall over all queries.
        QPS over all queries

    """
    if type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex):
        print(f"[FlatNav] searching with num-threads = {index.num_threads}")
        start = time.time()
        _, top_k_indices = index.search(
            queries=queries, ef_search=ef_search, K=k, num_initializations=300
        )
        end = time.time()
    else:
        print(f"[HNSW] search")
        index.set_ef(ef_search)
        start = time.time()
        # Search for HNSW return (ids, distances) instead of (distances, ids)
        top_k_indices, _ = index.knn_query(data=queries, k=k)
        end = time.time()

    querying_time = end - start
    qps = len(queries) / querying_time

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
