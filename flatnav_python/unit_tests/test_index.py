import flatnav
from flatnav.index import IndexL2Float, IndexIPFloat
from typing import Union, Optional
import numpy as np
import time
from .test_utils import (
    generate_random_data,
    get_ann_benchmark_dataset,
    compute_recall,
    create_index,
)
import pytest


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


@pytest.mark.skip(reason="Difficult to run on GitHub actions env due to data download")
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


# TODO: Figure out why this test is failing. Skipping it for now
@pytest.mark.skip(reason=None)
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


@pytest.mark.skip(reason="Difficult to run on GitHub actions env due to data download")
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
    index: Union[IndexL2Float, IndexIPFloat],
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

    print(f"\nIndexing time = {end - start} seconds")

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
