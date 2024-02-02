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
import plotly.express as px
from utils import compute_metrics

logging.basicConfig(level=logging.INFO)


ROOT_DATASET_PATH = "/root/data/"
# ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")

DATASET_NAMES = {
    "mnist-784-euclidean": "mnist",
    "sift-128-euclidean": "sift",
    "cauchy-10-euclidean": "cauchy10",
    "cauchy-256-euclidean": "cauchy256",
    "cauchy-1024-euclidean": "cauchy1024",
    "normal-10-angular": "normal10-cosine",
    "normal-256-angular": "normal256-cosine",
    "normal-1024-angular": "normal1024-cosine",
    "normal-10-euclidean": "normal10-l2",
    "normal-256-euclidean": "normal256-l2",
    "normal-1024-euclidean": "normal1024-l2",
}


def generate_iid_normal_dataset(
    num_samples: int,
    num_dimensions: int,
    num_queries: int,
    k: int,
    directory_path: str,
    dataset_name: str,
    metric: str = "cosine",
):
    """
    Generatest a dataset with the specified number of samples and dimensions using
    the standard normal distribution.
    Separates a subset for queries and computes their true k nearest neighbors.
    :param num_samples: Number of samples in the dataset.
    :param num_dimensions: Number of dimensions for each sample.
    :param num_queries: Number of queries to be separated from the dataset.
    :param k: The number of nearest neighbors to find.
    :param directory_path: Base path to save the dataset, queries, and ground truth labels.
    :param dataset_name: Name of the dataset (should be something like normal-10-angular)
    :param metric: Metric to use for computing nearest neighbors.
    """

    dataset = np.random.normal(size=(num_samples, num_dimensions))
    np.random.shuffle(dataset)
    query_set = dataset[:num_queries]
    dataset_without_queries = dataset[num_queries:]
    neighbors = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(
        dataset_without_queries
    )
    ground_truth_labels = neighbors.kneighbors(query_set, return_distance=False)

    # Normalize the dataset and queries if using cosine distance
    if metric in ["cosine", "angular", "ip"]:
        dataset_without_queries /= (
            np.linalg.norm(dataset_without_queries, axis=1, keepdims=True) + 1e-30
        )
        query_set /= np.linalg.norm(query_set, axis=1, keepdims=True) + 1e-30

    # Create directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Save the dataset
    np.save(
        f"{directory_path}/{dataset_name}.train.npy",
        dataset_without_queries.astype(np.float32),
    )
    np.save(f"{directory_path}/{dataset_name}.test.npy", query_set.astype(np.float32))
    np.save(
        f"{directory_path}/{dataset_name}.gtruth.npy",
        ground_truth_labels.astype(np.int32),
    )


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
    is_flatnav_index = type(index) in [flatnav.index.L2Index, flatnav.index.IPIndex]
    if is_flatnav_index:
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


def aggregate_metrics(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    k: int,
    num_initializations: int = 100,
) -> Tuple[dict, dict]:
    """
    Computes the following metrics for FlatNav and HNSW:
        - Recall@k
        - Latency
        - QPS
        - Hubness score as measured by the skewness of the k-occurence distribution (N_k)

    NOTE: Index construction is done in parallel, but search is single-threaded.

    :param train_dataset: The dataset to compute the skewness for.
    :param queries: The query vectors.
    :param ground_truth: The ground truth indices for each query.
    :param distance_type: The distance type to use for computing the skewness.
    :param max_edges_per_node: The maximum number of edges per node.
    :param ef_construction: The ef-construction parameter.
    :param ef_search: The ef-search parameter.
    :param k: The number of nearest neighbors to find.
    :param num_initializations: The number of initializations for FlatNav.
    """

    dataset_size, dim = train_dataset.shape
    hnsw_index = hnswlib.Index(
        space=distance_type if distance_type == "l2" else "ip", dim=dim
    )
    hnsw_index.init_index(
        max_elements=dataset_size,
        ef_construction=ef_construction,
        M=max_edges_per_node // 2,
    )

    hnsw_index.set_num_threads(os.cpu_count())

    logging.debug(f"Building index...")
    hnsw_base_layer_filename = "hnsw_base_layer.mtx"
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    hnsw_index.save_base_layer_graph(filename=hnsw_base_layer_filename)

    # Build FlatNav index
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        mtx_filename=hnsw_base_layer_filename,
    )

    # Here we will first allocate memory for the index and then build edge connectivity
    # using the HNSW base layer graph. We do not use the ef-construction parameter since
    # it's assumed to have been used when building the HNSW base layer.
    flatnav_index.allocate_nodes(data=train_dataset).build_graph_links()

    # Now delete the HNSW base layer graph since we don't need it anymore
    os.remove(hnsw_base_layer_filename)

    flatnav_index.set_num_threads(num_threads=1)
    hnsw_index.set_num_threads(num_threads=1)

    logging.debug(f"Computing skewness...")
    skewness_flatnav = compute_skewness(
        dataset=train_dataset, index=flatnav_index, ef_search=ef_search, k=k
    )
    skewness_hnsw = compute_skewness(
        dataset=train_dataset, index=hnsw_index, ef_search=ef_search, k=k
    )

    requested_metrics = [f"recall@{k}", "latency", "qps"]
    flatnav_metrics: dict = compute_metrics(
        requested_metrics=requested_metrics,
        index=flatnav_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )

    hnsw_metrics: dict = compute_metrics(
        requested_metrics=requested_metrics,
        index=hnsw_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )

    flatnav_metrics["skewness"] = skewness_flatnav
    hnsw_metrics["skewness"] = skewness_hnsw

    logging.info(f"FlatNav metrics: {flatnav_metrics}")
    logging.info(f"HNSW metrics: {hnsw_metrics}")

    return flatnav_metrics, hnsw_metrics


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
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Distance/metric to use (l2 or angular)",
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
            "Latency": metrics["latency_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )

    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Latency": metrics["latency_flatnav"],
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
        y="Latency",
        hue="Algorithm",
        style="Algorithm",
        data=df,
        s=100,
        ax=ax,
    )

    # Annotate each point with dataset name
    for i in range(len(df)):
        ax.text(
            df["Skewness"][i] + 0.5,
            df["Latency"][i] + 0.01,
            df["Dataset"][i],
            horizontalalignment="center",
            size="small",
            color="black",
            weight="normal",
        )

    sns.despine(trim=True, left=True)
    ax.set_title(f"Mean Query Latency vs Hubness score(skewness)")
    ax.legend()

    # Save the figure
    plt.savefig("hubness_seaborn.png")


def plot_metrics_plotly(metrics: dict, k: int):
    df_hnsw = pd.DataFrame(
        {
            "Skewness": metrics["skewness_hnsw"],
            "Latency": metrics["latency_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )
    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Latency": metrics["latency_flatnav"],
            "Algorithm": "FlatNav",
            "Dataset": metrics["dataset_names"],
        }
    )
    df = pd.concat([df_hnsw, df_flatnav], ignore_index=True)

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Skewness",
        y="Latency",
        color="Algorithm",
        symbol="Algorithm",
        size_max=15,
        hover_name="Dataset",  # Shows dataset name on hover
        title="Mean query latency vs hubness score",
    )
    fig.update_layout(
        legend_title_text="Algorithm",
        xaxis_title="Skewness",
        yaxis_title="Latency",
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="right", x=1),
    )
    fig.show()
    fig.write_html("hubness__.html")
    # fig.write_image("hubness.png")


if __name__ == "__main__":
    args = parse_args()

    # # Initialize a metrics dictionary to contain recall values for FlatNav and HNSW and the skewness values for FlatNav and HNSW
    metrics = {
        "latency_flatnav": [],
        "latency_hnsw": [],
        "skewness_flatnav": [],
        "skewness_hnsw": [],
        "dataset_names": [],
    }

    dataset_names = args.datasets
    distance_types = args.metrics
    if len(dataset_names) != len(distance_types):
        raise RuntimeError("Number of datasets and metrics/distances must be the same")

    for index, dataset_name in enumerate(dataset_names):
        metric = distance_types[index]
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        if not os.path.exists(base_path):
            # Create the directory if it doesn't exist
            os.makedirs(base_path)

            # Generate a dataset from IID standard normal
            dimension = int(dataset_name.split("-")[1])
            logging.info(f"Generating dataset with {dimension} dimensions")
            generate_iid_normal_dataset(
                num_samples=1010000,
                num_dimensions=dimension,
                num_queries=10000,
                k=args.k,
                directory_path=base_path,
                dataset_name=dataset_name,
            )
    train_dataset, queries, ground_truth = load_dataset(
        base_path=base_path, dataset_name=dataset_name
    )

    flatnav_metrics, hnsw_metrics = aggregate_metrics(
        train_dataset=train_dataset,
        queries=queries,
        ground_truth=ground_truth,
        distance_type=metric,
        max_edges_per_node=args.num_node_links,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        k=args.k,
    )

    metrics["latency_flatnav"].append(flatnav_metrics["latency"])
    metrics["latency_hnsw"].append(hnsw_metrics["latency"])
    metrics["skewness_flatnav"].append(flatnav_metrics["skewness"])
    metrics["skewness_hnsw"].append(hnsw_metrics["skewness"])
    metrics["dataset_names"].append(DATASET_NAMES[dataset_name])

    # Serialize metrics as JSON
    with open("hubness.json", "w") as f:
        json.dump(metrics, f)

    # read json file to metrics dictionary
    # with open("hubness.json", "r") as f:
    #     metrics = json.load(f)

    # Plot the metrics using seaborn
    plot_metrics_plotly(metrics=metrics, k=args.k)
    plot_metrics_seaborn(metrics=metrics, k=args.k)
