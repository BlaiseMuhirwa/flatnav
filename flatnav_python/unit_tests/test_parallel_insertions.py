import math
import time
import flatnav
from flatnav.index import index_factory
from flatnav.index import L2Index
from .test_utils import get_ann_benchmark_dataset, compute_recall, create_index, generate_random_data
import os
import numpy as np 

def test_parallel_insertions_yield_similar_recall():
    training_set, queries, ground_truth, _ = get_ann_benchmark_dataset(
        dataset_name="mnist-784-euclidean"
    )
    index = create_index(
        distance_type="l2",
        dim=training_set.shape[1],
        dataset_size=training_set.shape[0],
        max_edges_per_node=16,
    )
    single_threaded_index = create_index(
        distance_type="l2",
        dim=training_set.shape[1],
        dataset_size=training_set.shape[0],
        max_edges_per_node=16,
    )

    assert index.max_edges_per_node == 16

    index.num_threads = os.cpu_count()

    start = time.time()
    index.add(data=training_set, ef_construction=100)
    end = time.time()

    parallel_construction_time = end - start
    print(
        f"Index construction time (parallel): = {parallel_construction_time} seconds."
        f"Num-threads = {index.num_threads}"
    )

    recall_with_parallel_construction = compute_recall(
        index=index, queries=queries, ground_truth=ground_truth, ef_search=100
    )

    single_threaded_index.num_threads = 1

    start = time.time()
    single_threaded_index.add(data=training_set, ef_construction=100)
    end = time.time()

    single_threaded_index_construction_time = end - start
    print(
        f"Index construction time (single thread): = {single_threaded_index_construction_time}"
    )

    recall_with_single_threaded_index = compute_recall(
        index=index, queries=queries, ground_truth=ground_truth, ef_search=100
    )

    assert math.isclose(
        recall_with_parallel_construction,
        recall_with_single_threaded_index,
        abs_tol=1e-6,
    )

    # Construction time should be significantly lower for parallel insertions
    assert parallel_construction_time < single_threaded_index_construction_time
