import flatnav
from flatnav.index import index_factory
from flatnav.index import L2Index, IPIndex
from typing import Union, Optional
import numpy as np
import time
import tempfile
import h5py
import requests
import os


def generate_random_data(dataset_length: int, dim: int) -> np.ndarray:
    return np.random.rand(dataset_length, dim)


def get_ann_benchmark_dataset(dataset_name):
    base_uri = "http://ann-benchmarks.com"
    dataset_uri = f"{base_uri}/{dataset_name}.hdf5"

    with tempfile.TemporaryDirectory() as tmp:
        response = requests.get(dataset_uri)
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


def create_index(
    distance_type: str, dim: int, dataset_size: int, max_edges_per_node: int
) -> Union[L2Index, IPIndex]:
    index = index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
    )
    if not (isinstance(index, L2Index) or isinstance(index, IPIndex)):
        raise RuntimeError("Invalid index.")

    return index


def test_flatnav_l2_index_random_dataset():
    dataset_to_index = generate_random_data(dataset_length=30_000, dim=784)
    queries = generate_random_data(dataset_length=5_000, dim=784)
    ground_truth = np.random.randint(low=0, high=50, size=(5_000, 100))
    index = create_index(
        distance_type="l2",
        dim=dataset_to_index.shape[1],
        dataset_size=len(dataset_to_index),
        max_edges_per_node=32,
    )

    assert hasattr(index, "max_edges_per_node")
    assert index.max_edges_per_node == 32

    run_test(
        index=index,
        ef_construction=64,
        ef_search=32,
        training_set=dataset_to_index,
        queries=queries,
        ground_truth=ground_truth,
    )


def test_flatnav_l2_index_mnist_dataset():
    training_set, queries, ground_truth, _ = get_ann_benchmark_dataset(
        dataset_name="mnist-784-euclidean"
    )

    index = create_index(
        distance_type="l2",
        dim=training_set.shape[1],
        dataset_size=training_set.shape[0],
        max_edges_per_node=16,
    )

    assert hasattr(index, "max_edges_per_node")
    assert index.max_edges_per_node == 16

    run_test(
        index=index,
        ef_construction=128,
        ef_search=256,
        training_set=training_set,
        queries=queries,
        ground_truth=ground_truth,
        assert_recall_threshold=True,
        recall_threshold=0.97,
    )


def test_flatnav_ip_index_random_dataset():
    dataset_to_index = generate_random_data(dataset_length=30_000, dim=225)
    queries = generate_random_data(dataset_length=5_000, dim=225)
    ground_truth = np.random.randint(low=0, high=50, size=(5_000, 100))

    index = create_index(
        distance_type="angular",
        dim=dataset_to_index.shape[1],
        dataset_size=len(dataset_to_index),
        max_edges_per_node=16,
    )

    assert hasattr(index, "max_edges_per_node")
    assert index.max_edges_per_node == 16

    run_test(
        index=index,
        ef_construction=64,
        ef_search=32,
        training_set=dataset_to_index,
        queries=queries,
        ground_truth=ground_truth,
    )


def test_flatnav_index_with_reordering():
    training_set, queries, ground_truth, _ = get_ann_benchmark_dataset(
        dataset_name="mnist-784-euclidean"
    )

    index = create_index(
        distance_type="l2",
        dim=training_set.shape[1],
        dataset_size=training_set.shape[0],
        max_edges_per_node=16,
    )

    assert hasattr(index, "max_edges_per_node")
    assert index.max_edges_per_node == 16

    run_test(
        index=index,
        ef_construction=128,
        ef_search=256,
        training_set=training_set,
        queries=queries,
        ground_truth=ground_truth,
        assert_recall_threshold=True,
        recall_threshold=0.97,
        use_reordering=True,
        reordering_algorithm="gorder",
    )


def run_test(
    index: Union[L2Index, IPIndex],
    ef_construction: int,
    ef_search: int,
    training_set: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    use_reordering: bool = False,
    reordering_algorithm: Optional[str] = None,
    assert_recall_threshold: bool = False,
    recall_threshold: Optional[float] = None,
):
    start = time.time()
    index.add(data=training_set, ef_construction=ef_construction)
    end = time.time()

    print(f"Indexing time = {end - start} seconds")

    if use_reordering:
        if not reordering_algorithm:
            raise RuntimeError("Re-ordering algorithm must be provided.")
        index.reorder(algorithm=reordering_algorithm)

    recall = compute_recall(
        index=index, queries=queries, ground_truth=ground_truth, ef_search=ef_search
    )

    if assert_recall_threshold:
        if not recall_threshold:
            raise RuntimeError("Recall threshold must be provided.")
        assert recall >= recall_threshold
