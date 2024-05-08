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


# ROOT_DATASET_PATH = "/root/data/"
ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")

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

def depth_first_search(node: int, outdegree_table: np.ndarray, visited: set) -> None:
    visited.add(node)
    
    for neighbor_node in outdegree_table[node]:
        if neighbor_node not in visited:
            depth_first_search(node=neighbor_node, outdegree_table=outdegree_table, visited=visited)



def find_number_of_connected_components(outdegree_table: np.ndarray, subgraph: np.ndarray) -> int:
    """
    Returns the number of connected components in the subgraph.
    :param outdegree_table: The outdegree table for the graph. This graph is assumed to be directed (can be cyclic).
    :param subgraph: The subgraph to compute the number of connected components for.
        This will just be a list of node ids. The corresponding edges will be in the outdegree table.
    """
    
    visited = set()
    num_connected_components = 0
    for node in subgraph:
        if node not in visited:
            depth_first_search(node=node, outdegree_table=outdegree_table, visited=visited)
            num_connected_components += 1
            
    return num_connected_components
    
    
def plot_histogram(node_access_counts: dict) -> None:
    """
    Plots a histogram of the node access counts.
    :param node_access_counts: A dictionary mapping node ids to the number of times they were accessed.
    """
    
    # Compute the skewness of the node access counts
    node_access_counts = np.array(list(node_access_counts.values()))
    skewness = pd.Series(node_access_counts).skew()
    logging.info(f"Skewness of node access counts: {skewness}")
    
    # Plot the histogram
    plt.figure(figsize=(10, 10))
    sns.histplot(node_access_counts, bins=100, kde=True)
    plt.title(f"Node access counts (skewness: {skewness:.2f})")
    plt.xlabel("Number of times a node was accessed")
    plt.ylabel("Number of nodes")
    plt.show()





def aggregate_metrics(
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
) -> Tuple[dict, dict]:
    """
    Computes the following metrics for FlatNav and HNSW:
        - Recall@k
        - Latency
        - QPS
        - Hubness score as measured by the skewness of the k-occurence distribution (N_k)
        
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

    hnsw_index.set_num_threads(os.cpu_count())

    logging.debug(f"Building index...")
    hnsw_base_layer_filename = "hnsw_base_layer.mtx"
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    hnsw_index.save_base_layer_graph(filename=hnsw_base_layer_filename)

    # Build FlatNav index and configure it to perform search by using random initialization
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        mtx_filename=hnsw_base_layer_filename,
        use_random_initialization=True,
        random_seed=6771
    )

    # Here we will first allocate memory for the index and then build edge connectivity
    # using the HNSW base layer graph. We do not use the ef-construction parameter since
    # it's assumed to have been used when building the HNSW base layer.
    flatnav_index.allocate_nodes(data=train_dataset).build_graph_links()

    # Now delete the HNSW base layer graph since we don't need it anymore
    os.remove(hnsw_base_layer_filename)
    
    flatnav_index.set_num_threads(num_threads=1)
    hnsw_index.set_num_threads(num_threads=1)


    requested_metrics = [f"recall@{k}", "latency", "qps"]
    flatnav_metrics: dict = compute_metrics(
        requested_metrics=requested_metrics,
        index=flatnav_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )
    
    node_access_counts: dict = flatnav_index.get_node_access_counts()
    
    # Serialize the node access counts as JSON
    with open(f"{dataset_name}_node_access_counts.json", "w") as f:
        json.dump(node_access_counts, f)
    
    plot_histogram(node_access_counts)

    
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



if __name__ == "__main__":

    args = parse_args()

    dataset_names = args.datasets
    distance_types = args.metrics
    if len(dataset_names) != len(distance_types):
        raise RuntimeError("Number of datasets and metrics/distances must be the same")

    for index, dataset_name in enumerate(dataset_names):
        metric = distance_types[index]
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        if not os.path.exists(base_path):
            # Create the directory if it doesn't exist
            raise ValueError(f"Dataset path not found at {base_path}")
            
        train_dataset, queries, ground_truth = load_dataset(
            base_path=base_path, dataset_name=dataset_name
        )

        flatnav_metrics, hnsw_metrics = aggregate_metrics(
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

