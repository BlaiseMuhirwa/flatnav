import json
import pickle
import flatnav.index
import numpy as np
import argparse
import hnswlib
import os
from typing import Tuple, Union, List
import logging
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn 
from sklearn.neighbors import NearestNeighbors

# import plotly.express as px
from typing import List, Dict
from run_benchmark import compute_metrics

logging.basicConfig(level=logging.INFO)


ROOT_DATASET_PATH = "/root/data/hubness/data"
# ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")

# This should be a persistent volume mount.
DISTRIBUTIONS_SAVE_PATH = "/root/k-occurrence-dists"


SYNTHETIC_DATASETS = [
    "normal-16-angular",
    "normal-16-euclidean",
    "normal-32-angular",
    "normal-32-euclidean",
    "normal-64-angular",
    "normal-64-euclidean",
    "normal-128-angular",
    "normal-128-euclidean",
    "normal-256-angular",
    "normal-256-euclidean",
    "normal-1024-angular",
    "normal-1024-euclidean",
    "normal-1536-angular",
    "normal-1536-euclidean",
]

ANN_DATASETS = [
    "glove-100-angular",
    "nytimes-256-angular",
    "gist-960-euclidean",
    "yandex-deep-10m-euclidean",
    "spacev-10m-euclidean",
]


def main(
    dataset_name: str,
    train_dataset: np.ndarray,
    distance_type: str,
    k: int,
) -> list[int]:
    """
    Computes the k-occurence distribution for a given dataset.

    This distribution N_k is computed as follows:
    - Build a true Knn index with sklearn.
    - Sample 10k elements without replacement from the dataset.
    - For each element, find the true K nearest neighbors.
    - Count the number of times each element is a nearest neighbor.
    - Plot the distribution of counts.

    We yield a list of size 10k, where each element is the number of times
    that element was a nearest neighbor.
    """

    if not distance_type in ["cosine", "euclidean"]:
        raise RuntimeError("Invalid distance type")

    # Create the index
    sample_indices = np.random.choice(
        range(len(train_dataset)), size=10000, replace=False
    )
    index = NearestNeighbors(n_neighbors=k, metric=distance_type)
    index.fit(train_dataset[sample_indices])


    # Compute the k-nearest neighbors for each element in the sample
    distances, indices = index.kneighbors(train_dataset[sample_indices])

    # Count the number of times each sampled element is a nearest neighbor
    # in the returned indices. The k occurrence distribution must have the 
    # same size as the number of samples.
    k_occurrence_dist = np.zeros(len(sample_indices))

    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            k_occurrence_dist[neighbor] += 1

    return k_occurrence_dist

def get_metric_from_dataset_name(dataset_name: str) -> str:
    """
    Extract the metric from the dataset name. The metric is the last part of the dataset name.
    Ex. normal-10-euclidean -> l2
        mnist-784-euclidean -> l2
        normal-10-angular -> angular
    """
    metric = dataset_name.split("-")[-1]
    if metric == "euclidean":
        return "euclidean"
    elif metric == "angular":
        return "cosine"
    raise ValueError(f"Invalid metric: {metric}")


def load_dataset(base_path: str, dataset_name: str) -> Tuple[np.ndarray]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return (
        np.load(f"{base_path}/{dataset_name}.train.npy"),
        np.load(f"{base_path}/{dataset_name}.test.npy"),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy"),
    )

def run_main() -> None:
    datasets = ANN_DATASETS + SYNTHETIC_DATASETS

    for index, dataset_name in enumerate(datasets):

        print(f"Processing dataset {dataset_name}...")
        metric = get_metric_from_dataset_name(dataset_name)
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        if not os.path.exists(base_path):
            # Create the directory if it doesn't exist
            logging.error(f"Dataset path not found at {base_path}")

        train_dataset, queries, ground_truth = load_dataset(
            base_path=base_path, dataset_name=dataset_name
        )

        k_occurrence_dist = main(
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            distance_type=metric,
            k=100,
        )

        if not len(k_occurence_distribution) == 10000:
            raise RuntimeError(f"Invalid distribution size. Expected 10k, got {len(k_occurence_distribution)}")

        # Save this to disk 
        save_path = os.path.join(DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(k_occurrence_dist, f)



if __name__ == "__main__":
    run_main()
