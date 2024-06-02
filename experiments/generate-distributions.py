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
from scipy.stats import mannwhitneyu, ks_2samp

# import plotly.express as px
from typing import List, Dict

# import powerlaw
# from utils import compute_metrics
from run_benchmark import compute_metrics
from scipy.stats import skew

logging.basicConfig(level=logging.INFO)


ROOT_DATASET_PATH = "/root/data/"
# ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")

# This should be a persistent volume mount.
DISTRIBUTIONS_SAVE_PATH = "/root/node-access-distributions"
EDGE_DISTRIBUTIONS_SAVE_PATH = "/root/edge-lengths"


DATASET_NAMES = {
    "mnist-784-euclidean": "mnist",
    "sift-128-euclidean": "sift",
    "glove-100-angular": "glove",
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


class HubNodesConnectivityTester:
    """
    Class to test the connectivity of the hub nodes using a hypothesis test.
    Here is how the the test works:
    1. Hypotheses:
        - Null hypothesis (H0): Hub nodes are not more connected to each other than to randomly chosen nodes.
        - Alternative hypothesis (H1): Hub nodes are more connected to each other than to randomly chosen nodes.
    2. Identify hub nodes:
        - Select the nodes that fall into the 99th percentile of the node access counts.
    3. Define connectivity:
        - Presence of a direct edge in the graph.
    4. Calculate hub-hub connectivity:
        - For each hub node, calculate the number of hub nodes in its outdegree table.
        This yields a distribution of the number of hub nodes each hub node is connected to.
    5. Calculate random-hub connectivity:
        - For a set of randomly-chosen nodes (same size as the set of hub nodes), count
            the number of hub nodes in their outdegree table.
        This yields a distribution of the number of hub nodes each random node is connected to.
    6. Set up a statistical test:
        a.  - Use a one-sided Mann-Whitney U test to test the null hypothesis. The Mann-Whitney U test
            is a good option because it does not assume normality and it is non-parametric.
            - The null hypothesis is rejected if the p-value is less than 0.05.
        b.  - Use a one-sided Kolmogorov-Smirnov test to test the null hypothesis.
            - The null hypothesis is rejected if the p-value is less than 0.05.

    :param outdegree_table_path: The path to the pickle file containing the outdegree table.
    :param node_access_counts_path: The path to the pickle file containing the node access counts.

    """

    def __init__(
        self, outdegree_table_path: str, node_access_counts_path: str, dataset_name: str
    ):
        self.dataset_name = dataset_name
        self.outdegree_table = self._load_outdegree_table(outdegree_table_path)
        self.node_access_counts = self._load_node_access_counts(node_access_counts_path)

        print(f"length of outdegree table: {len(self.outdegree_table)}")
        print(f"length of node access counts: {len(self.node_access_counts)}")

    def _load_outdegree_table(self, outdegree_table_path: str) -> np.ndarray:
        """
        Read the outdegree table from the pickle file.
        """
        if not os.path.exists(outdegree_table_path):
            raise FileNotFoundError(
                f"Outdegree table not found at {outdegree_table_path}"
            )

        with open(outdegree_table_path, "rb") as f:
            return pickle.load(f)

    def _load_node_access_counts(self, node_access_counts_path: str) -> dict:
        """
        Read the node access counts from a JSON file.
        """
        if not os.path.exists(node_access_counts_path):
            raise FileNotFoundError(
                f"Node access counts not found at {node_access_counts_path}"
            )

        with open(node_access_counts_path, "r") as f:
            data = json.load(f)

        return {int(k): int(v) for k, v in data.items()}

    def _select_hub_nodes(self, percentile: float) -> List[int]:
        """
        Select the nodes that fall above the given percentile.
        :param percentile: The percentile to consider.
        :return selected_nodes: The subset of nodes that fall above the given percentile.
        """
        access_counts = list(self.node_access_counts.values())
        threshold = np.percentile(access_counts, percentile)
        selected_nodes = [
            node
            for node, count in self.node_access_counts.items()
            if count >= threshold
        ]
        return selected_nodes

    def _calculate_hub_hub_connections(self, hub_nodes: List[int]) -> List[int]:
        """
        Calculate the number of hub nodes in the outdegree table for each hub node.
        :param hub_nodes: The hub nodes to consider.
        :return hub_hub_connections: The number of hub nodes in the outdegree table for each hub node.
        """
        hub_hub_connections = []
        for node in hub_nodes:
            intersection = set(hub_nodes) & set(self.outdegree_table[node])
            hub_hub_connections.append(len(intersection))
        return hub_hub_connections

    def _calculate_random_hub_connections(
        self, hub_nodes: List[int], include_hub_nodes_in_sample: bool = False
    ) -> List[int]:
        """
        Calculate the number of hub nodes in the outdegree table for a set of randomly chosen nodes.
        :param hub_nodes: The hub nodes to consider.
        :param include_hub_nodes_in_sample: Whether to include the hub nodes in the sample of random nodes.
        :return random_hub_connectivity: The number of hub nodes in the outdegree table for each random node.
        """
        total_num_nodes = len(self.node_access_counts)
        num_hub_nodes = len(hub_nodes)
        if num_hub_nodes >= total_num_nodes:
            raise ValueError(
                "Number of hub nodes must be less than the total number of nodes."
            )

        # Sample without replacement n nodes from the graph.
        if include_hub_nodes_in_sample:
            random_nodes = np.random.choice(
                total_num_nodes, size=num_hub_nodes, replace=False
            )
        else:
            non_hub_nodes = set(self.node_access_counts.keys()) - set(hub_nodes)
            if len(non_hub_nodes) < num_hub_nodes:
                raise ValueError(
                    "Not enough non-hub nodes to sample from. "
                    f"Num-hub nodes = {num_hub_nodes}, total nodes = {total_num_nodes}"
                )
            random_nodes = np.random.choice(
                non_hub_nodes, size=num_hub_nodes, replace=False
            )

        random_hub_connectivity = []
        for node in random_nodes:
            intersection = set(hub_nodes) & set(self.outdegree_table[node])
            random_hub_connectivity.append(len(intersection))
        return random_hub_connectivity

    def run_hypothesis_test(self, percentile: float) -> None:
        """
        Run the hypothesis test to determine if hub nodes are more connected to each
        other than to randomly chosen nodes.
        :param percentile: The percentile to consider.
        """
        hub_nodes = self._select_hub_nodes(percentile=percentile)
        logging.info(f"Num hub nodes = {len(hub_nodes)}, dataset = {self.dataset_name}")

        hub_hub_connections: List[int] = self._calculate_hub_hub_connections(
            hub_nodes=hub_nodes
        )

        random_hub_connections: List[int] = self._calculate_random_hub_connections(
            hub_nodes=hub_nodes
        )
        if len(hub_hub_connections) != len(random_hub_connections):
            raise ValueError(
                "Hub-hub connections and random-hub connections must have the same length."
            )

        # Perform the statistical tests
        # Mann-Whitney U test
        u_statistic, p_value = mannwhitneyu(
            hub_hub_connections, random_hub_connections, alternative="greater"
        )
        logging.info(f"Mann-Whitney U test: p-value = {p_value}")

        # Kolmogorov-Smirnov test
        ks_statistic, p_value = ks_2samp(
            hub_hub_connections, random_hub_connections, alternative="greater"
        )
        logging.info(f"Kolmogorov-Smirnov test: p-value = {p_value}")


def depth_first_search(node: int, outdegree_table: np.ndarray, visited: set) -> None:
    visited.add(node)

    for neighbor_node in outdegree_table[node]:
        if neighbor_node not in visited:
            depth_first_search(
                node=neighbor_node, outdegree_table=outdegree_table, visited=visited
            )


def find_number_of_connected_components(
    outdegree_table: np.ndarray, subgraph: np.ndarray
) -> int:
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
            depth_first_search(
                node=node, outdegree_table=outdegree_table, visited=visited
            )
            num_connected_components += 1

    return num_connected_components


def plot_histogram(node_access_counts: dict, dataset_name: str) -> None:
    """
    Plots a histogram of the node access counts.
    """

    # Convert the dictionary values to a numpy array
    counts = np.array(list(node_access_counts.values()))

    # Ensure data integrity
    assert np.all(np.isfinite(counts)), "Data contains non-finite values."
    assert np.all(counts > 0), "Data contains non-positive values."

    # Compute the skewness of the node access counts
    skewness = pd.Series(counts).skew()
    logging.info(f"Skewness of node access counts: {skewness}")

    # Plot setup
    plt.figure(figsize=(10, 10))

    # Calculate logarithmically spaced bins
    bins = np.logspace(np.log10(1), np.log10(10000), 60)

    # Plotting the histogram using matplotlib
    plt.hist(counts, bins=bins, log=True, edgecolor="black")
    plt.xscale("log")  # Ensuring the x-axis is log-scaled

    # Titles and labels
    plt.title(
        f"Dataset name: {dataset_name} -- Node access counts (skewness: {skewness:.4f})"
    )
    plt.xlabel("Node access counts, N")
    plt.ylabel("Frequency")

    # Manually setting the x-ticks to handle log scale ticks more appropriately
    plt.xticks([10, 100, 1000, 10000], labels=["10", "100", "1,000", "10,000"])

    # Save the figure
    figurename = f"{DISTRIBUTIONS_SAVE_PATH}/{dataset_name}_node_access_counts.png"
    plt.savefig(figurename)

    logging.info(f"Saved figure at {figurename}")


def get_node_access_counts_distribution(
    dataset_name: str,
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex],
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

    # name = DATASET_NAMES[dataset_name]
    # plot_pmf_from_node_accesses(node_access_counts=node_access_counts, dataset_name=name)
    # plot_histogram(node_access_counts=node_access_counts, dataset_name=name)
    # fit_power_law(distribution=node_access_counts, save_path=f"{name}-powerlaw.png")


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


def plot_edge_length_distribution(distribution: dict, dataset_name: str) -> None:
    """
    Plots a histogram of the edge lengths.
    """
    # Normalize the distribution by the largest edge length
    # so that the histogram is easier to interpret
    # max_edge_length = max(distribution.values())
    # distribution = {k: v / max_edge_length for k, v in distribution.items()}

    skewness = pd.Series(distribution.values()).skew()
    bins = np.logspace(
        np.log10(min(distribution.values())),
        np.log10(max(distribution.values())),
        num=60,
    )

    # Plot the histogram
    plt.figure(figsize=(10, 10))
    sns.histplot(distribution.values(), bins=bins, kde=False, color="blue")
    plt.title(f"{dataset_name} Edge length distribution (skewness: {skewness:.4f})")
    plt.xlabel("Edge length")
    plt.ylabel("Number of edges")
    plt.savefig(f"{EDGE_DISTRIBUTIONS_SAVE_PATH}/{dataset_name}_edge_lengths.png")


def select_p90_nodes(
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


def get_edge_length_distribution_for_hubs(index, dataset_name: str):
    node_access_dist_file = os.path.join(
        DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json"
    )
    with open(node_access_dist_file, "r") as f:
        node_access_counts = json.load(f)

    # Convert keys and values to integers
    node_access_counts = {int(k): int(v) for k, v in node_access_counts.items()}

    hub_nodes = select_p90_nodes(node_access_counts, 99)
    logging.info(f"Number of hub nodes = {len(hub_nodes)}")

    distribution: dict = index.get_edge_length_distribution_for_nodes(
        node_ids=hub_nodes
    )

    json_filepath = os.path.join(
        EDGE_DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_hub_edge_lengths.json"
    )
    with open(json_filepath, "w") as f:
        json.dump(distribution, f)


def get_edge_lengths_distribution(
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex], dataset_name: str
):
    # This is a dictionary mapping unique edge hashes to their lengths.
    # The lengths are floating point numbers.
    # We want to plot a histogram of this distribution.
    distribution: dict = index.get_edge_length_distribution()

    # Save the distribution as JSON file
    json_filepath = os.path.join(
        EDGE_DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_edge_lengths.json"
    )
    with open(json_filepath, "w") as f:
        json.dump(distribution, f)

    # # Normalize the distribution by the largest edge length
    # # so that the histogram is easier to interpret
    # max_edge_length = max(distribution.values())
    # distribution = {k: v / max_edge_length for k, v in distribution.items()}

    # skewness = pd.Series(distribution.values()).skew()

    # # Plot the histogram
    # plt.figure(figsize=(10, 10))
    # sns.histplot(distribution.values(), bins="auto", kde=True, log_scale=(False, True))
    # plt.title(f"{dataset_name} Edge length distribution (skewness: {skewness:.4f})")
    # plt.xlabel("Edge length")
    # plt.ylabel("Number of edges")

    # filename = os.path.join(EDGE_DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_edge_lengths.png")
    # plt.savefig(filename)

    # logging.info(f"Plot saved at {filename}")


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
    distribution_type: str,
    num_initializations: int = 100,
) -> Tuple[dict, dict]:
    """
    Computes either the node access counts distribution or the edge lengths distribution for a given dataset.

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
    :param distribution_type: The desired distribution to consider. Can be either 'node-access' or 'edge-length'.
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

    start = time.time()
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    end = time.time()
    logging.info(f"Indexing time = {end - start} seconds")

    mtx_filename = "hnsw_index.mtx"
    hnsw_index.save_base_layer_graph(filename=mtx_filename)

    # Build FlatNav index and configure it to perform search by using random initialization
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=True,
        use_random_initialization=True,
        random_seed=42,
    )

    flatnav_index.allocate_nodes(train_dataset).build_graph_links(mtx_filename)
    os.remove(mtx_filename)
    flatnav_index.set_num_threads(1)

    # # Train the index.
    # start = time.time()
    # flatnav_index.add(
    #     data=train_dataset,
    #     ef_construction=ef_construction,
    #     num_initializations=num_initializations,
    # )
    # end = time.time()
    # logging.info(f"Index construction time: {end - start:.3f} seconds")

    if distribution_type.lower() == "node-access":
        node_access_counts = get_node_access_counts_distribution(
            dataset_name=dataset_name,
            index=flatnav_index,
            queries=queries,
            ground_truth=ground_truth,
            ef_search=ef_search,
            k=k,
        )

        return node_access_counts

    elif distribution_type.lower() == "edge-length":
        name = dataset_name.replace("euclidean", "l2").replace("angular", "cosine")
        get_edge_length_distribution_for_hubs(index=flatnav_index, dataset_name=name)
    else:
        raise ValueError(f"Invalid distribution type: {distribution_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        nargs="+",
        help="dataset names. All will be expected to be at the same path.",
    )

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

    parser.add_argument(
        "--distribution-type",
        type=str,
        required=True,
        help="Desired distribution to consider. Can be either 'node-access' or 'edge-length'.",
    )

    return parser.parse_args()


def load_dataset(base_path: str, dataset_name: str) -> Tuple[np.ndarray]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return (
        np.load(f"{base_path}/{dataset_name}.train.npy"),
        np.load(f"{base_path}/{dataset_name}.test.npy"),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy"),
    )


def get_metric_from_dataset_name(dataset_name: str) -> str:
    """
    Extract the metric from the dataset name. The metric is the last part of the dataset name.
    Ex. normal-10-euclidean -> l2
        mnist-784-euclidean -> l2
        normal-10-angular -> angular
    """
    metric = dataset_name.split("-")[-1]
    if metric == "euclidean":
        return "l2"
    elif metric == "angular":
        return "angular"
    raise ValueError(f"Invalid metric: {metric}")


def run_main(args: argparse.Namespace) -> None:
    dataset_names = args.datasets

    # Map from dataset name to node access counts
    distributions = {}

    for index, dataset_name in enumerate(dataset_names):

        print(f"Processing dataset {dataset_name}...")
        metric = get_metric_from_dataset_name(dataset_name)
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        if not os.path.exists(base_path):
            # Create the directory if it doesn't exist
            raise ValueError(f"Dataset path not found at {base_path}")

        train_dataset, queries, ground_truth = load_dataset(
            base_path=base_path, dataset_name=dataset_name
        )

        main(
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            queries=queries,
            ground_truth=ground_truth,
            distance_type=metric,
            max_edges_per_node=args.num_node_links,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            k=args.k,
            distribution_type=args.distribution_type,
        )

        # Save the node access counts for this dataset to a JSON file
        # filepath = os.path.join(DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json")
        # with open(filepath, "w") as f:
        #     json.dump(node_access_counts, f)

        # Load the node access counts from the JSON file
        dataset_name = dataset_name.replace("euclidean", "l2").replace(
            "angular", "cosine"
        )
        filepath = os.path.join(
            EDGE_DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_edge_lengths.json"
        )
        with open(filepath, "r") as f:
            edge_length_distribution = json.load(f)

        # Plot the edge length distribution
        plot_edge_length_distribution(
            distribution=edge_length_distribution, dataset_name=dataset_name
        )

        # distributions[dataset_name] = node_access_counts

        # plot_histogram(node_access_counts=node_access_counts, dataset_name=dataset_name)

    # # Now plot the distributions
    # bw_adjust_values = [0.7]
    # for bw_adjust_value in bw_adjust_values:
    #     plot_kde_distributions(distributions, DISTRIBUTIONS_SAVE_PATH, bw_adjust_value)


if __name__ == "__main__":
    args = parse_args()

    dataset_names = args.datasets
    for dataset_name in dataset_names:
        outdegree_table_path = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_outdegree_table.pkl"
        )
        node_access_counts_path = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json"
        )

        tester = HubNodesConnectivityTester(
            outdegree_table_path=outdegree_table_path,
            node_access_counts_path=node_access_counts_path,
            dataset_name=dataset_name,
        )
        tester.run_hypothesis_test(percentile=99)

