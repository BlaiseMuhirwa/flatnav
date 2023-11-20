import flatnav
from flatnav.index import index_factory
from flatnav.index import L2Index, IPIndex
from typing import Union
import pytest
import numpy as np
import time 


def generate_random_data(dataset_length: int, dim: int) -> np.ndarray:
    return np.random.rand(dataset_length, dim)


def create_index(
    distance_type: str, dim: int, dataset_size: int, max_edges_per_node: int
) -> Union[L2Index, IPIndex]:
    index = index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True
    )
    if not (
        isinstance(index, flatnav.index.L2Index)
        or isinstance(index, flatnav.index.IPIndex)
    ):
        raise RuntimeError("Invalid index.")

    return index


def test_flatnav_l2_index():
    dataset_to_index = generate_random_data(dataset_length=60_000, dim=784)
    queries = generate_random_data(dataset_length=10_000, dim=784)
    index = create_index(
        distance_type="l2",
        dim=dataset_to_index.shape[1],
        dataset_size=len(dataset_to_index),
        max_edges_per_node=32,
    )

    assert hasattr(index, "max_edges_per_node")
    assert index.max_edges_per_node == 32

    start = time.time()
    index.add(data=dataset_to_index, ef_construction=64)
    end = time.time()
    
    print(f"Indexing time = {end - start}")

    
    start = time.time()
    distances, node_ids = index.search(queries=queries, ef_search=64, K=100)
    end = time.time()
    print(f"Querying time = {end - start}")

    assert distances.shape == node_ids.shape


"""
Indexing time = 693.3694415092468
Querying time = 48.112215518951416
"""