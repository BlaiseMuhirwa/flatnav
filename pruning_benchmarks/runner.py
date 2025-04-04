import time
import json
import flatnav
from flatnav import PruningHeuristic
from flatnav.data_type import DataType
from flatnav import BuildParameters, MemoryAllocator
import numpy as np
from typing import Optional, Tuple, List, Dict, Union, Any
import os
import logging
import argparse
from data_loader import get_data_loader
from metrics import metric_manager


# --- Define Experiments To Run ---
# List of dictionaries, each specifying an experiment configuration.
# 'name': Used for logging and JSON keys. Should be unique.
# 'base_heuristic': The PruningHeuristic enum value.
# 'parameter': The float parameter value, or None if not applicable.
# 'alpha': Specific alpha for Vamana variants (optional). Let's assume
#          the C++ handles VAMANA default alpha and lower alpha via parameter for now.
#          If alpha needs to be set separately, the C++/Python interface needs adjustment.
#          For simplicity here, we'll represent VAMANA variations using the parameter field.
#          parameter=-1.0 for VAMANA_LOWER_ALPHA is one way, C++ needs to interpret it.
#          parameter=-2.0 for CHEAP_OUTDEGREE_CONDITIONAL_M
#          parameter=-3.0 for CHEAP_OUTDEGREE_CONDITIONAL_EDGE_THRESHOLD

EXPERIMENTS_TO_RUN: List[Dict[str, Any]] = [
    # Arya Mount Family
    {
        "name": "arya_mount",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT,
        "parameter": None,
    },
    {
        "name": "arya_mount_sanity_check",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_SANITY_CHECK,
        "parameter": None,
    },
    {
        "name": "arya_mount_reversed",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_REVERSED,
        "parameter": None,
    },
    {
        "name": "arya_mount_shuffled",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_SHUFFLED,
        "parameter": None,
    },
    {
        "name": "arya_mount_plus_spanner",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_PLUS_SPANNER,
        "parameter": None,
    },
    # Vamana Family (Assuming parameter distinguishes variants or C++ handles default)
    {
        "name": "vamana_alpha_1.2",
        "base_heuristic": PruningHeuristic.VAMANA,
        "parameter": 1.2,
    },  # Example: pass desired alpha
    {
        "name": "vamana_alpha_0.83",
        "base_heuristic": PruningHeuristic.VAMANA,
        "parameter": 0.8333,
    },  # Example: pass lower alpha
    # Simple KNN
    {
        "name": "nearest_m",
        "base_heuristic": PruningHeuristic.NEAREST_M,
        "parameter": None,
    },
    {
        "name": "furthest_m",
        "base_heuristic": PruningHeuristic.FURTHEST_M,
        "parameter": None,
    },
    # Adaptive / Baseline
    {
        "name": "median_adaptive",
        "base_heuristic": PruningHeuristic.MEDIAN_ADAPTIVE,
        "parameter": None,
    },
    {
        "name": "top_m_median_adaptive",
        "base_heuristic": PruningHeuristic.TOP_M_MEDIAN_ADAPTIVE,
        "parameter": None,
    },
    {
        "name": "mean_sorted_baseline",
        "base_heuristic": PruningHeuristic.MEAN_SORTED_BASELINE,
        "parameter": None,
    },
    {
        "name": "geometric_mean",
        "base_heuristic": PruningHeuristic.GEOMETRIC_MEAN,
        "parameter": None,
    },
    # Parameterized (Single Base Enum)
    {
        "name": "quantile_not_min_0.2",
        "base_heuristic": PruningHeuristic.QUANTILE_NOT_MIN,
        "parameter": 0.2,
    },
    {
        "name": "probabilistic_rank_1.0",
        "base_heuristic": PruningHeuristic.PROBABILISTIC_RANK,
        "parameter": 1.0,
    },
    {
        "name": "neighborhood_overlap_0.8",
        "base_heuristic": PruningHeuristic.NEIGHBORHOOD_OVERLAP,
        "parameter": 0.8,
    },
    {
        "name": "sigmoid_ratio_1.0",
        "base_heuristic": PruningHeuristic.SIGMOID_RATIO,
        "parameter": 1.0,
    },
    {
        "name": "sigmoid_ratio_5.0",
        "base_heuristic": PruningHeuristic.SIGMOID_RATIO,
        "parameter": 5.0,
    },
    {
        "name": "sigmoid_ratio_10.0",
        "base_heuristic": PruningHeuristic.SIGMOID_RATIO,
        "parameter": 10.0,
    },
    {
        "name": "am_random_rejects_0.01",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_RANDOM_ON_REJECTS,
        "parameter": 0.01,
    },
    {
        "name": "am_random_rejects_0.05",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_RANDOM_ON_REJECTS,
        "parameter": 0.05,
    },
    {
        "name": "am_random_rejects_0.10",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_RANDOM_ON_REJECTS,
        "parameter": 0.10,
    },
    {
        "name": "am_sigmoid_rejects_0.1",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_SIGMOID_ON_REJECTS,
        "parameter": 0.1,
    },
    {
        "name": "am_sigmoid_rejects_5.0",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_SIGMOID_ON_REJECTS,
        "parameter": 5.0,
    },
    {
        "name": "am_sigmoid_rejects_10.0",
        "base_heuristic": PruningHeuristic.ARYA_MOUNT_SIGMOID_ON_REJECTS,
        "parameter": 10.0,
    },
    # Cheap Outdegree Conditional (Using Parameter for threshold)
    # Special Values: -3.0 = Default Edge Threshold, -2.0 = M
    {
        "name": "cheap_outdegree_thresh_default",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": -3.0,
    },
    {
        "name": "cheap_outdegree_thresh_M",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": -2.0,
    },
    {
        "name": "cheap_outdegree_thresh_2",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 2.0,
    },
    {
        "name": "cheap_outdegree_thresh_4",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 4.0,
    },
    {
        "name": "cheap_outdegree_thresh_6",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 6.0,
    },
    {
        "name": "cheap_outdegree_thresh_8",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 8.0,
    },
    {
        "name": "cheap_outdegree_thresh_10",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 10.0,
    },
    {
        "name": "cheap_outdegree_thresh_12",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 12.0,
    },
    {
        "name": "cheap_outdegree_thresh_14",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 14.0,
    },
    {
        "name": "cheap_outdegree_thresh_16",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 16.0,
    },
    {
        "name": "cheap_outdegree_thresh_20",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 20.0,
    },
    {
        "name": "cheap_outdegree_thresh_24",
        "base_heuristic": PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL,
        "parameter": 24.0,
    },
    # Large Outdegree Conditional (Assuming C++ uses local calc or takes parameter)
    # Let's assume it takes a parameter like Cheap Outdegree for now
    # Special Value: -3.0 = Default Well Connected Threshold
    {
        "name": "large_outdegree_thresh_default",
        "base_heuristic": PruningHeuristic.LARGE_OUTDEGREE_CONDITIONAL,
        "parameter": -3.0,
    },
    # Other
    {
        "name": "one_spanner",
        "base_heuristic": PruningHeuristic.ONE_SPANNER,
        "parameter": None,
    },
]


def compute_metrics(
    requested_metrics: List[str],
    index: Union[flatnav.index.IndexL2Float, flatnav.index.IndexIPFloat],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k=100,
) -> Dict[str, float]:
    latencies = []
    top_k_indices = []
    distance_computations = []
    index.get_query_distance_computations()
    for query in queries:
        start = time.time()
        _, indices = index.search_single(query=query, ef_search=ef_search, K=k)
        end = time.time()
        latencies.append(end - start)
        top_k_indices.append(indices)
        query_dist_computations = index.get_query_distance_computations()
        distance_computations.append(query_dist_computations)

    querying_time = sum(latencies)
    total_distance_computations = sum(distance_computations)
    num_queries = len(queries)
    kwargs = {
        "querying_time": querying_time,
        "num_queries": num_queries,
        "latencies": latencies,
        "distance_computations_list": distance_computations,
        "total_distance_computations": total_distance_computations,
        "queries": queries,
        "ground_truth": ground_truth,
        "top_k_indices": top_k_indices,
        "k": k,
    }
    metrics = {}
    for metric_name in requested_metrics:
        try:
            if metric_name in metric_manager.metric_functions:
                metric_kwargs = kwargs.copy()
                # Adjust kwargs if specific metrics need different inputs
                metrics[metric_name] = metric_manager.compute_metric(
                    metric_name, **metric_kwargs
                )
        except Exception:
            logging.error(f"Error computing metric {metric_name}", exc_info=True)
    return metrics


def train_index(
    train_dataset: np.ndarray,
    distance_type: str,
    dim: int,
    dataset_size: int,
    max_edges_per_node: int, 
    ef_construction: int,
    base_heuristic: PruningHeuristic,
    parameter_value: Optional[float],
    num_build_threads: int = 1,
) -> Tuple[
    Union[flatnav.index.IndexL2Float, flatnav.index.IndexIPFloat],
    MemoryAllocator,
    float,
]:
    """
    Creates and trains a FlatNav index on the given dataset using a specific
    base pruning heuristic and its parameter.
    """
    heuristic_name_str = str(base_heuristic).split(".")[-1]
    param_str = f" (Param: {parameter_value})" if parameter_value is not None else ""
    logging.info(
        f"Building FlatNav index: M={max_edges_per_node}, efC={ef_construction}, "
        f"Heuristic={heuristic_name_str}{param_str}, Threads={num_build_threads}"
    )

    # Create index configuration and pre-allocate memory
    params = BuildParameters(
        dim=dim,
        M=max_edges_per_node,
        dataset_size=dataset_size,
        data_type=DataType.float32,
        ef_construction=ef_construction,
        pruning_heuristic=base_heuristic,
        # Pass the parameter value (can be None)
        pruning_heuristic_parameter=parameter_value,
    )
    allocator = MemoryAllocator(params=params)

    # Create the index instance
    index = flatnav.index.create(
        distance_type=distance_type,
        params=params,
        mem_allocator=allocator,
        verbose=True,
        collect_stats=True,
    )

    index.set_num_threads(num_build_threads)

    start = time.time()
    index.add(data=train_dataset)
    end = time.time()
    indexing_time = end - start
    logging.info(f"Indexing time = {indexing_time:.4f} seconds")

    return index, allocator, indexing_time


def main(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    gtruth: np.ndarray,
    ef_cons_params: List[int],
    ef_search_params: List[int],
    num_node_links: List[int],
    distance_type: str,
    metrics_file: str,
    dataset_name: str,
    requested_metrics: List[str],
    k_neighbors: int,
    num_build_threads: int = 1,
    num_search_threads: int = 1,
):
    dataset_size = train_dataset.shape[0]
    dim = train_dataset.shape[1]

    all_metrics_data = {}
    if os.path.exists(metrics_file) and os.path.getsize(metrics_file) > 0:
        try:
            with open(metrics_file, "r") as file:
                all_metrics_data = json.load(file)
                logging.info(f"Loaded existing metrics from {metrics_file}")
        except json.JSONDecodeError:
            logging.error(f"Error reading existing {metrics_file}. Starting fresh.")
            all_metrics_data = {}


    def build_and_run_search(
        ef_cons: int,
        node_links: int,
        base_heuristic_enum: PruningHeuristic,
        parameter_value: Optional[float],
        heuristic_display_name: str, 
    ):
        """
        Builds the index, runs KNN search for all ef_search values, and collects metrics.
        Manages index/allocator lifetime to control memory.
        """
        nonlocal all_metrics_data

        index = None
        allocator = None
        log_heuristic_name = heuristic_display_name

        try:
            index, allocator, indexing_time = train_index(
                train_dataset=train_dataset,
                max_edges_per_node=node_links,
                ef_construction=ef_cons,
                dataset_size=dataset_size,
                dim=dim,
                distance_type=distance_type,
                base_heuristic=base_heuristic_enum,
                parameter_value=parameter_value,
                num_build_threads=num_build_threads,
            )

            index.set_num_threads(num_search_threads)
            experiment_base_key = f"{dataset_name}_{distance_type}_{log_heuristic_name}_M{node_links}_efC{ef_cons}"

            if experiment_base_key not in all_metrics_data:
                all_metrics_data[experiment_base_key] = {"params": {}, "runs": []}

            all_metrics_data[experiment_base_key]["params"] = {
                "dataset": dataset_name,
                "distance": distance_type,
                "heuristic_name": log_heuristic_name,
                "base_heuristic": str(base_heuristic_enum), 
                "parameter": parameter_value,
                "M": node_links,
                "ef_construction": ef_cons,
                "build_threads": num_build_threads,
                "search_threads": num_search_threads,
                "k": k_neighbors,
                "dataset_size": dataset_size,
                "dim": dim,
                "indexing_time_sec": indexing_time,
            }

            existing_runs = {
                run["ef_search"]: run
                for run in all_metrics_data[experiment_base_key]["runs"]
            }

            for ef_search in ef_search_params:
                if ef_search in existing_runs:
                    logging.warning(
                        f"Skipping ef_search={ef_search} for {experiment_base_key}, already exists."
                    )
                    continue

                logging.info(
                    f"Running search: ef_search={ef_search} for {experiment_base_key}"
                )
                start_compute_time = time.time()
                current_metrics = compute_metrics(
                    requested_metrics=requested_metrics,
                    index=index,
                    queries=queries,
                    ground_truth=gtruth,
                    ef_search=ef_search,
                    k=k_neighbors,
                )
                end_compute_time = time.time()
                logging.info(f"Metrics (ef_search={ef_search}): {current_metrics}")
                logging.info(
                    f"Metric computation time: {end_compute_time - start_compute_time:.4f} seconds"
                )

                current_metrics["ef_search"] = ef_search
                all_metrics_data[experiment_base_key]["runs"].append(current_metrics)

                # Incremental save
                try:
                    with open(metrics_file, "w") as file:
                        json.dump(all_metrics_data, file, indent=4)
                except Exception as e:
                    logging.error(
                        f"Failed to write metrics incrementally to {metrics_file}: {e}"
                    )

        except Exception as e:
            logging.error(
                f"Failed experiment run for efC={ef_cons}, M={node_links}, Heuristic={log_heuristic_name}",
                exc_info=True,
            )
        finally:
            logging.debug(
                f"Deleting index and allocator for efC={ef_cons}, M={node_links}, Heuristic={log_heuristic_name}"
            )
            del index
            del allocator
            import gc
            gc.collect()



    for experiment_config in EXPERIMENTS_TO_RUN:
        base_heuristic = experiment_config["base_heuristic"]
        parameter = experiment_config["parameter"]
        display_name = experiment_config["name"]

        for node_links in num_node_links:
            for ef_cons in ef_cons_params:
                logging.info(
                    f"--- Starting Config: Heuristic='{display_name}', M={node_links}, efC={ef_cons} ---"
                )
                build_and_run_search(
                    ef_cons=ef_cons,
                    node_links=node_links,
                    base_heuristic_enum=base_heuristic,
                    parameter_value=parameter,
                    heuristic_display_name=display_name,
                )

    # Final save
    try:
        with open(metrics_file, "w") as file:
            json.dump(all_metrics_data, file, indent=4)
        logging.info(f"Final metrics saved to {metrics_file}")
    except Exception as e:
        logging.error(f"Failed to write final metrics to {metrics_file}: {e}")


# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FlatNav Pruning Heuristics (Parameterized)."
    )

    # --- Data Arguments ---
    parser.add_argument(
        "--dataset", required=True, help="Path to the training dataset (.npy file)."
    )
    parser.add_argument(
        "--queries", required=True, help="Path to the queries file (.npy file)."
    )
    parser.add_argument(
        "--gtruth",
        required=True,
        help="Path to the ground truth file (.npy file, containing indices).",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        type=str,
        help="Name of the benchmark dataset (e.g., 'sift1m').",
    )
    parser.add_argument(
        "--train-dataset-range",
        required=False,
        default=None,
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Load a slice of the training dataset [START, END).",
    )

    # --- Index Parameters ---
    parser.add_argument(
        "--metric", required=True, choices=["l2", "angular"], help="Distance metric."
    )
    parser.add_argument(
        "--num-node-links",
        nargs="+",
        type=int,
        default=[16],
        help="List of M values (max edges per node).",
    )
    parser.add_argument(
        "--ef-construction",
        nargs="+",
        type=int,
        default=[100],
        help="List of efConstruction values.",
    )

    # --- Search Parameters ---
    parser.add_argument(
        "--ef-search",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500],
        help="List of efSearch values.",
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=100, help="Number of nearest neighbors (k)."
    )

    # --- Performance/System Parameters ---
    parser.add_argument(
        "--num-build-threads",
        default=1,
        type=int,
        help="Threads for index construction.",
    )
    parser.add_argument(
        "--num-search-threads", default=1, type=int, help="Threads for search."
    )

    # --- Output and Metrics ---
    parser.add_argument(
        "--requested-metrics",
        required=False,
        nargs="+",
        type=str,
        default=[
            "recall",
            "qps",
            "latency_p50",
            "avg_dist_comps_per_query",
        ],
        help="List of metrics to compute and store.",
    )
    parser.add_argument(
        "--metrics-file",
        required=False,
        default="pruning_heuristics_metrics.json",
        help="File to save results (JSON).",
    )
    parser.add_argument(
        "--metrics-dir",
        required=False,
        default="pruning_metrics",
        help="Directory for metrics file and plots.",
    )

    return parser.parse_args()



def run_experiment():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()

    metrics_dir = args.metrics_dir
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_file_path = os.path.join(metrics_dir, args.metrics_file)
    logging.info(f"Metrics will be saved to: {metrics_file_path}")

    logging.info(
        f"Loading data: Dataset='{args.dataset}', Queries='{args.queries}', GTruth='{args.gtruth}'"
    )
    data_loader = get_data_loader(
        train_dataset_path=args.dataset,
        queries_path=args.queries,
        ground_truth_path=args.gtruth,
        range=args.train_dataset_range,
    )
    train_data, queries, ground_truth = data_loader.load_data()

    if ground_truth.shape[1] < args.k_neighbors:
        logging.warning(
            f"Ground truth has {ground_truth.shape[1]} neighbors, k={args.k_neighbors}. Recall capped."
        )

    main(
        train_dataset=train_data,
        queries=queries,
        gtruth=ground_truth,
        ef_cons_params=args.ef_construction,
        ef_search_params=args.ef_search,
        num_node_links=args.num_node_links,
        distance_type=args.metric.lower(),
        dataset_name=args.dataset_name,
        num_build_threads=args.num_build_threads,
        num_search_threads=args.num_search_threads,
        metrics_file=metrics_file_path,
        k_neighbors=args.k_neighbors,
        requested_metrics=args.requested_metrics,
    )
    logging.info("Experiment finished.")


if __name__ == "__main__":
    run_experiment()
