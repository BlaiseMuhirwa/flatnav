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
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind, skew
from utils import get_metric_from_dataset_name, load_dataset

# import plotly.express as px
from typing import List, Dict

# import powerlaw
# from utils import compute_metrics
from run_benchmark import compute_metrics

logging.basicConfig(level=logging.INFO)


ROOT_DATASET_PATH = "/root/data/hubness/data"
# ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")

# This should be a persistent volume mount.
DISTRIBUTIONS_SAVE_PATH = "/root/node-access-distributions"


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


def get_node_access_counts_distribution(
    dataset_name: str,
    index: "flanav.index.Index",
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k: int,
):
    """
    Computes the node access counts distribution for a given index.
    :param index: The index to compute the node access counts distribution for.
    :param queries: The query vectors.
    :param ground_truth: The ground truth indices for each query.
    :param ef_search: The ef-search parameter.
    :param k: The number of nearest neighbors to find.
    """

    index.set_num_threads(1)
    requested_metrics = [f"recall", "latency_p50", "qps"]
    flatnav_metrics: dict = compute_metrics(
        requested_metrics=requested_metrics,
        index=index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )

    logging.info(f"FlatNav metrics: {flatnav_metrics}")

    node_access_counts: dict = index.get_node_access_counts()
    return node_access_counts


def plot_kde_distributions(distributions, save_path, bw_adjust_value=0.3):
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()  # Get the current Axes instance

    # Compute skewness before the log transformation and store them
    skewness_values = {}
    for dataset_name, node_access_counts in distributions.items():
        skewness_values[dataset_name] = skew(list(node_access_counts.values()))

    for dataset_name, node_access_counts in distributions.items():
        # Apply log-transform to the counts, adding 1 to avoid log(0)
        log_counts = np.log1p(list(node_access_counts.values()))
        raw_skewness = skewness_values[dataset_name]
        # Replace "euclidean" with "l2" and "angular" with "cosine"
        dataset_name = dataset_name.replace("euclidean", "l2").replace(
            "angular", "cosine"
        )

        # Plot the KDE for log-transformed data with less smoothness
        sns.kdeplot(
            log_counts,
            label=f"{dataset_name} ($\\tilde{{\\mu}}_3$ = {raw_skewness:.4f})",
            bw_adjust=bw_adjust_value,
        )

    # Set up the legend on the right of the plot
    plt.legend(title="Dataset", loc="center left", bbox_to_anchor=(1, 0.5))

    # Improve plot aesthetics
    sns.despine(trim=True)  # Trim the spines for a cleaner look
    plt.grid(True)  # Add gridlines
    plt.xlabel("Log of Node access counts")
    plt.ylabel("PDF")
    plt.title("KDE of Node Access Counts")

    # Adjust the plot area to fit the legend and increase the resolution
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()

    filename = os.path.join(save_path, f"distributions_{bw_adjust_value}.png")
    plt.savefig(filename)


def select_nodes_based_on_percentile(
    node_access_counts: Dict[int, int], percentile: float
) -> List[int]:
    """
    Select the nodes that fall above the 90th percentile.
    :param node_access_counts: The node access counts.
    :param percentile: The percentile to consider.
    :return selected_nodes: The subset of nodes that fall above the 90th percentile.
    """
    access_counts = list(node_access_counts.values())
    threshold = np.percentile(access_counts, percentile)
    selected_nodes = [
        node for node, count in node_access_counts.items() if count >= threshold
    ]
    return selected_nodes

def main(
    dataset_name: str,
    train_dataset: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    k: int,
    num_initializations: int = 100,
) -> Tuple[dict, list]:
    """
    Computes the outdegree table and the node access distribution for a given dataset.

    NOTE: Index construction is done in parallel, but search is single-threaded.

    :param dataset_name: The name of the dataset.
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

    hnsw_index.set_num_threads(32)

    start = time.time()
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    end = time.time()
    logging.info(f"Indexing time = {end - start} seconds")

    mtx_filename = "hnsw_index.mtx"
    hnsw_index.save_base_layer_graph(filename=mtx_filename)

    # Build FlatNav index and configure it to perform search by using random initialization
    # We do this here so that there is no preferential treatment for certain nodes during search.
    # We want to compute the access 
    flatnav_index = flatnav.index.create(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=False,
        use_random_initialization=True,
        random_seed=42,
    )

    flatnav_index.allocate_nodes(train_dataset).build_graph_links(mtx_filename)
    os.remove(mtx_filename)
    flatnav_index.set_num_threads(1)

    node_access_counts: dict[int, int] = get_node_access_counts_distribution(
        dataset_name=dataset_name,
        index=flatnav_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )

    outdegree_table: list[list[int]] = flatnav_index.get_graph_outdegree_table()

    return node_access_counts, outdegree_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k",
        type=int,
        required=False,
        default=100,
        help="Number of nearest neighbors to consider",
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



def run_main(args: argparse.Namespace) -> None:
    datasets = ANN_DATASETS + SYNTHETIC_DATASETS
    # Map from dataset name to node access counts
    distributions = {}

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

        node_access_counts, outdegree_table = main(
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            queries=queries,
            ground_truth=ground_truth,
            distance_type=metric,
            max_edges_per_node=args.num_node_links,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            k=args.k,
        )

        if len(node_access_counts.keys()) != len(train_dataset):
            logging.error(
                f"Node access counts length mismatch: {len(node_access_counts)} vs {len(train_dataset)}"
            )

        if len(outdegree_table) != len(train_dataset):
            logging.error(
                f"Outdegree table length mismatch: {len(outdegree_table)} vs {len(train_dataset)}"
            )

        # Save the node access counts for this dataset to a JSON file
        filepath = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json"
        )
        with open(filepath, "w") as f:
            json.dump(node_access_counts, f)

        # Save the outdegree table for this dataset to a pickle file
        outdegree_table_path = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_outdegree_table.pkl"
        )
        with open(outdegree_table_path, "wb") as f:
            pickle.dump(outdegree_table, f)


    # # Now plot the distributions
    # bw_adjust_values = [0.7]
    # for bw_adjust_value in bw_adjust_values:
    #     plot_kde_distributions(distributions, DISTRIBUTIONS_SAVE_PATH, bw_adjust_value)


if __name__ == "__main__":
    args = parse_args()
    run_main(args=args)
