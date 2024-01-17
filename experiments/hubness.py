import json
import flatnav.index
import numpy as np
import argparse
import os
from typing import Tuple, Union
import logging
import hnswlib
import time
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import compute_metrics

logging.basicConfig(level=logging.INFO)


# ROOT_DATASET_PATH = "/root/data/"
ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")

DATASET_NAMES = {
    "mnist-784-euclidean": "mnist",
    "sift-128-euclidean": "sift",
    "cauchy-10-euclidean": "cauchy10",
    "cauchy-256-euclidean": "cauchy256",
    "cauchy-1024-euclidean": "cauchy1024",
}


def generate_cauchy_dataset(
    num_samples: int,
    num_dimensions: int,
    num_queries: int,
    k: int,
    base_path: str,
    metric: str = "minkowski",
    p: int = 2,
):
    """
    Generates a dataset with the specified number of samples and dimensions using
    the Cauchy distribution.
    Separates a subset for queries and computes their true k nearest neighbors.

    :param num_samples: Number of samples in the dataset.
    :param num_dimensions: Number of dimensions for each sample.
    :param num_queries: Number of queries to be separated from the dataset.
    :param k: The number of nearest neighbors to find.
    :param base_path: Base path to save the dataset, queries, and ground truth labels.
    :param metric: Metric to use for computing nearest neighbors.
    :param p: Parameter for the metric.

    NOTE: metric="minkowski" and p=2 is equivalent to Euclidean distance.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    # Generate the dataset
    dataset = np.random.standard_cauchy(size=(num_samples, num_dimensions))

    # Separate out a subset for queries
    np.random.shuffle(dataset)
    query_set = dataset[:num_queries]
    dataset_without_queries = dataset[num_queries:]

    # Compute the true k nearest neighbors for the query set
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute", p=p, metric=metric).fit(
        dataset_without_queries
    )
    ground_truth_labels = nbrs.kneighbors(query_set, return_distance=False)

    # Save the dataset without queries, the queries, and the ground truth labels
    np.save(f"{base_path}/train.npy", dataset_without_queries.astype(np.float32))
    np.save(f"{base_path}/test.npy", query_set.astype(np.float32))
    np.save(f"{base_path}/ground_truth.npy", ground_truth_labels.astype(np.int32))


def compute_k_occurence_distrubution(top_k_indices: np.ndarray) -> np.ndarray:
    """
    Computes the distribution of k-occurences for each node in the given array.
    :param top_k_indices: array of shape (dataset_size, k) containing the indices of
            the k nearest neighbors for each node.

    :return: array of shape (dataset_size,) containing the k-occurence distribution for each node (N_k)
    """

    # validate indices. If any value is negative, throw an error
    if np.any(top_k_indices < 0):
        raise ValueError("Indices cannot be negative")

    dataset_size = top_k_indices.shape[0]
    k_occurence_distribution = np.zeros(dataset_size, dtype=int)

    flattened_indices = top_k_indices.flatten()
    unique_indices, counts = np.unique(flattened_indices, return_counts=True)
    k_occurence_distribution[unique_indices] = counts

    return k_occurence_distribution


def compute_skewness(
    index: Union[flatnav.index.L2Index, hnswlib.Index],
    dataset: np.ndarray,
    ef_search: int,
    k: int,
) -> float:
    if type(index) == flatnav.index.L2Index:
        _, top_k_indices = index.search(
            queries=dataset,
            ef_search=ef_search,
            K=k,
        )
    elif type(index) == hnswlib.Index:
        top_k_indices, _ = index.knn_query(dataset, k=k)
    else:
        raise ValueError("Invalid index")

    k_occurence_distribution = compute_k_occurence_distrubution(
        top_k_indices=top_k_indices
    )
    mean = np.mean(k_occurence_distribution)
    std_dev = np.std(k_occurence_distribution)
    denominator = len(k_occurence_distribution) * (std_dev**3)
    skewness = (np.sum((k_occurence_distribution - mean) ** 3)) / denominator

    return skewness


def get_recall_and_dataset_skewness(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    k: int,
    num_initializations: int = 100,
):
    dataset_size, dim = train_dataset.shape
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=False,
    )
    hnsw_index = hnswlib.Index(space=distance_type, dim=dim)
    hnsw_index.init_index(
        max_elements=dataset_size,
        ef_construction=ef_construction,
        M=max_edges_per_node // 2,
    )

    flatnav_index.set_num_threads(os.cpu_count())
    hnsw_index.set_num_threads(os.cpu_count())

    logging.debug(f"Building index...")
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    flatnav_index.add(
        data=train_dataset,
        ef_construction=ef_construction,
        num_initializations=num_initializations,
    )

    # We currently have a bufferoverlow issue during FlatNav's multithreaded search, so
    # I'm setting the number of threads to 1 for now until it's fixed.
    flatnav_index.set_num_threads(num_threads=1)

    skewness_flatnav = compute_skewness(
        dataset=train_dataset, index=flatnav_index, ef_search=ef_search, k=k
    )
    skewness_hnsw = compute_skewness(
        dataset=train_dataset, index=hnsw_index, ef_search=ef_search, k=k
    )

    logging.info(f"Skewness: Flatnav: {skewness_flatnav}, HNSW: {skewness_hnsw}")

    recall_flatnav, _ = compute_metrics(
        index=flatnav_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )
    recall_hnsw, _ = compute_metrics(
        index=hnsw_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )
    logging.info(f"Recall@{k}: Flatnav: {recall_flatnav}, HNSW: {recall_hnsw}")

    return recall_flatnav, recall_hnsw, skewness_flatnav, skewness_hnsw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute hubness of a dataset")

    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        nargs="+",
        help="dataset names. All will be expected to be at theh same path.",
    )

    parser.add_argument(
        "--k",
        type=int,
        required=False,
        default=100,
        help="Number of nearest neighbors to consider",
    )
    parser.add_argument(
        "--metric", type=str, required=True, help="Metric to use (l2 or angular)"
    )
    parser.add_argument(
        "--ef-construction", type=int, required=True, help="ef-construction parameter."
    )

    parser.add_argument(
        "--ef-search", type=int, required=True, help="ef-search parameter."
    )

    parser.add_argument(
        "--num-node-links",
        type=int,
        required=True,
        help="max-edges-per-node parameter.",
    )

    return parser.parse_args()


def load_dataset(base_path: str, dataset_name: str) -> Tuple[np.ndarray]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")

    return (
        np.load(f"{base_path}/{dataset_name}.train.npy").astype(np.float32),
        np.load(f"{base_path}/{dataset_name}.test.npy").astype(np.float32),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy").astype(np.uint32),
    )


def plot_metrics_seaborn(metrics: dict, k: int):
    df_hnsw = pd.DataFrame(
        {
            "Skewness": metrics["skewness_hnsw"],
            "Recall": metrics["recall_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )

    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Recall": metrics["recall_flatnav"],
            "Algorithm": "FlatNav",
            "Dataset": metrics["dataset_names"],
        }
    )

    # Combine both DataFrames into one for plotting
    df = pd.concat([df_hnsw, df_flatnav], ignore_index=True)

    # Set the style and context for the plot
    sns.set(style="whitegrid", context="talk")
    f, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x="Skewness",
        y=f"Recall@{k}",
        hue="Algorithm",
        style="Algorithm",
        data=df,
        s=100,
        ax=ax,
    )

    # Annotate each point with dataset name
    for i in range(len(df)):
        ax.text(
            df["Skewness"][i],
            df["Recall"][i],
            df["Dataset"][i],
            horizontalalignment="center",
            size="medium",
            color="black",
            weight="semibold",
        )

    sns.despine(trim=True, left=True)
    ax.set_title(f"Comparison of Skewness and Recall@{k} between HNSW and FlatNav")
    ax.legend(frameon=True)

    # Save the figure
    plt.savefig("hubness_seaborn.png")


if __name__ == "__main__":
    args = parse_args()

    # Initialize a metrics dictionary to contain recall values for FlatNav and HNSW and the skewness values for FlatNav and HNSW
    metrics = {
        "recall_flatnav": [],
        "recall_hnsw": [],
        "skewness_flatnav": [],
        "skewness_hnsw": [],
        "dataset_names": [],
    }

    dataset_names = args.datasets
    for dataset_name in dataset_names:
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)
        train_dataset, queries, ground_truth = load_dataset(
            base_path=base_path, dataset_name=dataset_name
        )
        (
            recall_flatnav,
            recall_hnsw,
            skewness_flatnav,
            skewness_hnsw,
        ) = get_recall_and_dataset_skewness(
            train_dataset=train_dataset,
            queries=queries,
            ground_truth=ground_truth,
            distance_type=args.metric,
            max_edges_per_node=args.num_node_links,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            k=args.k,
        )

        # Aggregate metrics
        metrics["recall_flatnav"].append(recall_flatnav)
        metrics["recall_hnsw"].append(recall_hnsw)
        metrics["skewness_flatnav"].append(skewness_flatnav)
        metrics["skewness_hnsw"].append(skewness_hnsw)
        metrics["dataset_names"].append(DATASET_NAMES[dataset_name])

    # Serialize metrics as JSON
    with open("hubness.json", "w") as f:
        json.dump(metrics, f)

    plot_metrics_seaborn(metrics=metrics, k=args.k)
