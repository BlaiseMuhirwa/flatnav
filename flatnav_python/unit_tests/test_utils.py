from typing import Union, Optional
import numpy as np
import tempfile
import h5py
import requests
import os
import time
import flatnav
from flatnav.index import IndexL2Float, IndexIPFloat, create


def create_index(
    distance_type: str, dim: int, dataset_size: int, max_edges_per_node: int
) -> Union[IndexL2Float, IndexIPFloat]:
    index = create(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
    )
    if not (isinstance(index, IndexL2Float) or isinstance(index, IndexIPFloat)):
        raise RuntimeError("Invalid index.")

    return index


def generate_random_data(dataset_length: int, dim: int) -> np.ndarray:
    return np.random.rand(dataset_length, dim)


def get_ann_benchmark_dataset(dataset_name):
    base_uri = "http://ann-benchmarks.com"
    dataset_uri = f"{base_uri}/{dataset_name}.hdf5"

    with tempfile.TemporaryDirectory() as tmp:
        response = requests.get(dataset_uri, timeout=120)
        loc = os.path.join(tmp, dataset_name)

        with open(loc, "wb") as f:
            f.write(response.content)
        data = h5py.File(loc, "r")

    training_set = data["train"]
    queries = data["test"]
    true_neighbors = data["neighbors"]
    distances = data["distances"]

    return (
        np.array(training_set),
        np.array(queries),
        np.array(true_neighbors),
        np.array(distances),
    )


def compute_recall(
    index, queries: np.ndarray, ground_truth: np.ndarray, ef_search: int, k: int = 100
):
    """
    Compute recall for given queries, ground truth, and a FlatNav index.

    Args:
        - index: The Faiss index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.

    Returns:
        Mean recall over all queries.
    """
    start = time.time()
    _, top_k_indices = index.search(queries=queries, ef_search=ef_search, K=k)
    end = time.time()

    duration = (end - start) / len(queries)
    print(f"Querying time: {duration * 1000} milliseconds")

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)
    return recall
