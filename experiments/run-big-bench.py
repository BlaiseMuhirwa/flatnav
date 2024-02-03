import time
from typing import Union
import json
import hnswlib
import numpy as np
from typing import Optional, Tuple, List
import numpy as np
from dvclive import Live
import os
import logging
import platform, socket, psutil
import argparse
import flatnav


ENVIRONMENT_INFO = {
    "load_before_experiment": os.getloadavg()[2],
    "platform": platform.platform(),
    "platform_version": platform.version(),
    "platform_release": platform.release(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "hostname": socket.gethostname(),
    "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
    "num_cores": psutil.cpu_count(logical=True),
}


def load_sift_dataset(
    train_dataset_path: str, queries_path: str, gtruth_path: str
) -> Tuple[np.ndarray]:
    return (
        np.load(train_dataset_path).astype(np.float32),
        np.load(queries_path).astype(np.float32),
        np.load(gtruth_path).astype(np.uint32),
    )


def load_benchmark_dataset(
    train_dataset_path: str,
    queries_path: str,
    gtruth_path: str,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray]:
    def verify_paths_exist(paths: List[str]) -> None:
        for path in paths:
            if not os.path.exists(path):
                raise ValueError(f"Invalid file path: {path}")

    def load_ground_truth(path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Load the IDs and the distances of the top-k's and not the distances.
        Returns:
            - Array of top k IDs
            - Array of top k distances
            - Number of queries
            - K value
        """
        with open(path, "rb") as f:
            num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
            K = np.fromfile(f, dtype=np.uint32, count=1)[0]

        # Memory-map the IDs only
        ground_truth_ids = np.memmap(
            path,
            dtype=np.uint32,
            mode="r",
            shape=(num_queries, K),
            offset=8,
        )

        ground_truth_dists = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            shape=(num_queries, K),
            offset=8 + (num_queries * K * np.dtype(np.uint32).itemsize),
        )

        return ground_truth_ids, ground_truth_dists, num_queries, K

    verify_paths_exist([train_dataset_path, queries_path, gtruth_path])

    train_dtype = np.float32 if train_dataset_path.endswith("fbin") else np.uint8
    total_size = os.path.getsize(train_dataset_path) // np.dtype(train_dtype).itemsize

    # Read header information (num_points and num_dimensions)
    with open(train_dataset_path, "rb") as f:
        num_points = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]

    if chunk_size:
        bytes_to_load = chunk_size * num_dimensions * np.dtype(train_dtype).itemsize
        train_dataset = np.memmap(
            train_dataset_path,
            dtype=train_dtype,
            mode="r",
            shape=(total_size - 2,),
            offset=8,
        )
        train_dataset = train_dataset[: bytes_to_load // np.dtype(train_dtype).itemsize]
        train_dataset = train_dataset.reshape((chunk_size, num_dimensions))
    else:
        train_dataset = np.fromfile(train_dataset_path, dtype=train_dtype, offset=8)
        train_dataset = train_dataset.reshape((num_points, num_dimensions))

    gtruth_dataset, _, num_queries, _ = load_ground_truth(gtruth_path)
    queries_dataset = np.fromfile(
        queries_path,
        dtype=np.float32 if queries_path.endswith("fbin") else np.uint8,
        offset=8,
    )
    queries_dataset = queries_dataset.reshape((num_queries, num_dimensions))

    return train_dataset, queries_dataset, gtruth_dataset


def compute_metrics(
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex, hnswlib.Index],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k=100,
) -> Tuple[float, float]:
    """
    Compute recall and QPS for given queries, ground truth for the given index(FlatNav or HNSW).

    Args:
        - index: A FlatNav index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.

    Returns:
        Mean recall over all queries.
        QPS over all queries

    """
    is_flatnav_index = type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex)
    if is_flatnav_index:
        index.set_num_threads(1)
        start = time.time()
        _, top_k_indices = index.search(
            queries=queries, ef_search=ef_search, K=k, num_initializations=300
        )
        end = time.time()
    else:
        index.set_num_threads(1)
        index.set_ef(ef_search)
        start = time.time()
        # Search for HNSW return (ids, distances) instead of (distances, ids)
        top_k_indices, _ = index.knn_query(data=queries, k=k)
        end = time.time()

    querying_time = end - start
    qps = len(queries) / querying_time

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)

    return recall, qps


def create_and_train_hnsw_index(
    data: np.ndarray,
    space: str,
    dim: int,
    dataset_size: int,
    ef_construction: int,
    max_edges_per_node: int,
    num_threads,
) -> hnswlib.Index:
    hnsw_index = hnswlib.Index(space=space, dim=dim)
    hnsw_index.init_index(
        max_elements=dataset_size, ef_construction=ef_construction, M=max_edges_per_node
    )
    hnsw_index.set_num_threads(num_threads)

    start = time.time()
    hnsw_index.add_items(data=data, ids=np.arange(dataset_size))
    end = time.time()
    logging.info(f"Indexing time = {end - start} seconds")

    return hnsw_index


def train_index(
    train_dataset: np.ndarray,
    distance_type: str,
    dim: int,
    dataset_size: int,
    max_edges_per_node: int,
    ef_construction: int,
    index_type: str = "flatnav",
    use_hnsw_base_layer: bool = False,
    hnsw_base_layer_filename: Optional[str] = None,
    num_build_threads: int = 1,
) -> Union[flatnav.index.L2Index, flatnav.index.IPIndex, hnswlib.Index]:
    if index_type == "hnsw":
        # We use "angular" instead of "ip", so here we are just converting.
        _distance_type = distance_type if distance_type == "l2" else "ip"
        # HNSWlib will have M * 2 edges in the base layer.
        # So if we want to use M=32, we need to set M=16 here.
        hnsw_index = create_and_train_hnsw_index(
            data=train_dataset,
            space=_distance_type,
            dim=dim,
            dataset_size=dataset_size,
            ef_construction=ef_construction,
            max_edges_per_node=max_edges_per_node // 2,
            num_threads=num_build_threads,
        )

        return hnsw_index

    if use_hnsw_base_layer:
        if not hnsw_base_layer_filename:
            raise ValueError("Must provide a filename for the HNSW base layer graph.")

        _distance_type = distance_type if distance_type == "l2" else "ip"
        hnsw_index = create_and_train_hnsw_index(
            data=train_dataset,
            space=_distance_type,
            dim=dim,
            dataset_size=dataset_size,
            ef_construction=ef_construction,
            max_edges_per_node=max_edges_per_node // 2,
            num_threads=num_build_threads,
        )

        # Now extract the base layer's graph and save it to a file.
        # This will be a Matrix Market file that we use to construct the Flatnav index.
        hnsw_index.save_base_layer_graph(filename=hnsw_base_layer_filename)

        index = flatnav.index.index_factory(
            distance_type=distance_type,
            dim=dim,
            mtx_filename=hnsw_base_layer_filename,
        )

        # Here we will first allocate memory for the index and then build edge connectivity
        # using the HNSW base layer graph. We do not use the ef-construction parameter since
        # it's assumed to have been used when building the HNSW base layer.
        index.allocate_nodes(data=train_dataset).build_graph_links()

    else:
        index = flatnav.index.index_factory(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            verbose=False,
        )
        index.set_num_threads(num_build_threads)

        # Train the index.
        start = time.time()
        index.add(
            data=train_dataset, ef_construction=ef_construction, num_initializations=300
        )
        end = time.time()

        logging.info(f"Indexing time = {end - start} seconds")

    return index


def main(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    gtruth: np.ndarray,
    ef_cons_params: List[int],
    ef_search_params: List[int],
    num_node_links: List[int],
    distance_type: str,
    index_type: str = "flatnav",
    use_hnsw_base_layer: bool = False,
    hnsw_base_layer_filename: Optional[str] = None,
    reordering_strategies: List[str] | None = None,
    num_build_threads: int = 1,
    num_search_threads: int = 1,
):
    dataset_size = train_dataset.shape[0]
    dim = train_dataset.shape[1]

    for node_links in num_node_links:
        for ef_cons in ef_cons_params:
            for ef_search in ef_search_params:
                index = train_index(
                    index_type=index_type,
                    train_dataset=train_dataset,
                    max_edges_per_node=node_links,
                    ef_construction=ef_cons,
                    dataset_size=dataset_size,
                    dim=dim,
                    distance_type=distance_type,
                    use_hnsw_base_layer=use_hnsw_base_layer,
                    hnsw_base_layer_filename=hnsw_base_layer_filename,
                    num_build_threads=num_build_threads,
                )

                if reordering_strategies is not None:
                    if type(index) not in (
                        flatnav.index.L2Index,
                        flatnav.index.IPIndex,
                    ):
                        raise ValueError(
                            "Reordering strategies only apply to FlatNav index."
                        )
                    index.reorder(strategies=reordering_strategies)

                if num_search_threads > 1:
                    index.set_num_threads(num_search_threads)

                recall, qps = compute_metrics(
                    index=index,
                    queries=queries,
                    ground_truth=gtruth,
                    ef_search=ef_search,
                )

                logging.info(
                    f"Recall@100: {recall}, QPS={qps}, node_links={node_links},"
                    f" ef_cons={ef_cons}, ef_search={ef_search}"
                )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Flatnav on Big ANN datasets."
    )

    parser.add_argument(
        "--index-type",
        default="flatnav",
        help="Type of index to benchmark. Options include `flatnav` and `hnsw`.",
    )

    parser.add_argument(
        "--use-hnsw-base-layer",
        action="store_true",
        help="If set, use HNSW's base layer's connectivity for the Flatnav index.",
    )
    parser.add_argument(
        "--hnsw-base-layer-filename",
        default=None,
        help="Filename to save the HNSW base layer graph to. Please use the .mtx extension for clarity.",
    )

    parser.add_argument(
        "--num-node-links",
        nargs="+",
        type=int,
        default=[16, 32],
        help="Number of node links per node.",
    )

    parser.add_argument(
        "--ef-construction",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500],
        help="ef_construction parameter.",
    )

    parser.add_argument(
        "--ef-search",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500, 1000, 2000, 3000, 4000],
        help="ef_search parameter.",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a single ANNS benchmark dataset to run on.",
    )
    parser.add_argument(
        "--queries", required=True, help="Path to a singe queries file."
    )
    parser.add_argument(
        "--gtruth",
        required=True,
        help="Path to a single ground truth file to evaluate on.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        default="l2",
        help="Distance tye. Options include `l2` and `angular`.",
    )

    parser.add_argument(
        "--reordering-strategies",
        required=False,
        nargs="+",
        type=str,
        default=None,
        help="A sequence of graph re-ordering strategies(only applies to FlatNav index)."
        "Options include `gorder` and `rcm`.",
    )

    parser.add_argument(
        "--num-build-threads",
        required=False,
        default=1,
        type=int,
        help="Number of threads to use during index construction.",
    )

    parser.add_argument(
        "--num-search-threads",
        required=False,
        default=1,
        type=int,
        help="Number of threads to use during search.",
    )

    parser.add_argument(
        "--log_metrics", required=False, default=False, help="Log metrics to DVC."
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args = parse_arguments()

    train_data, queries, ground_truth = load_benchmark_dataset(
        train_dataset_path=args.dataset,
        queries_path=args.queries,
        gtruth_path=args.gtruth,
    )
    main(
        train_dataset=train_data,
        queries=queries,
        gtruth=ground_truth,
        ef_cons_params=args.ef_construction,
        ef_search_params=args.ef_search,
        num_node_links=args.num_node_links,
        distance_type=args.metric.lower(),
        index_type=args.index_type.lower(),
        use_hnsw_base_layer=args.use_hnsw_base_layer,
        hnsw_base_layer_filename=args.hnsw_base_layer_filename,
        reordering_strategies=args.reordering_strategies,
        num_build_threads=args.num_build_threads,
        num_search_threads=args.num_search_threads,
    )
