import numpy as np
import flatnav
from flatnav.utils import PruningHeuristic
from flatnav.data_type import DataType
import time
from flatnav import BuildParameters
from flatnav import MemoryAllocator

ALL_HEURISTICS = [
    PruningHeuristic.ARYA_MOUNT,
    PruningHeuristic.VAMANA,
    PruningHeuristic.VAMANA_LOWER_ALPHA,
    PruningHeuristic.ARYA_MOUNT_SANITY_CHECK,
    PruningHeuristic.NEAREST_M,
    PruningHeuristic.FURTHEST_M,
    PruningHeuristic.MEDIAN_ADAPTIVE,
    PruningHeuristic.TOP_M_MEDIAN_ADAPTIVE,
    PruningHeuristic.MEAN_SORTED_BASELINE,
    PruningHeuristic.QUANTILE_NOT_MIN,
    PruningHeuristic.ARYA_MOUNT_REVERSED,
    PruningHeuristic.PROBABILISTIC_RANK,
    PruningHeuristic.NEIGHBORHOOD_OVERLAP,
    PruningHeuristic.GEOMETRIC_MEAN,
    PruningHeuristic.SIGMOID_RATIO_STEEPNESS_1,
    PruningHeuristic.SIGMOID_RATIO_STEEPNESS_5,
    PruningHeuristic.SIGMOID_RATIO_STEEPNESS_10,
    PruningHeuristic.ARYA_MOUNT_SHUFFLED,
    PruningHeuristic.ARYA_MOUNT_RANDOM_ON_REJECTS,
    PruningHeuristic.ARYA_MOUNT_RANDOM_ON_REJECTS_5,
    PruningHeuristic.ARYA_MOUNT_RANDOM_ON_REJECTS_10,
    PruningHeuristic.ARYA_MOUNT_SIGMOID_ON_REJECTS,
    PruningHeuristic.ARYA_MOUNT_SIGMOID_ON_REJECTS_STEEPNESS_5,
    PruningHeuristic.ARYA_MOUNT_SIGMOID_ON_REJECTS_STEEPNESS_10,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_EDGE_THRESHOLD,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_2,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_4,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_6,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_8,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_10,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_12,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_14,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_16,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_20,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_24,
    PruningHeuristic.CHEAP_OUTDEGREE_CONDITIONAL_M,
    PruningHeuristic.LARGE_OUTDEGREE_CONDITIONAL,
    PruningHeuristic.ONE_SPANNER,
    PruningHeuristic.ARYA_MOUNT_PLUS_SPANNER,
]


def compute_recall(queries, ground_truth, top_k_indices, k) -> float:
    ground_truth_sets = [set(gt) for gt in ground_truth]
    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)
    return recall


def compute_metrics(
    index,
    queries,
    ground_truth,
    ef_searches,
    k=100,
):
    # Returns an {ef_search_value: {metric_name: metric_value}}.
    output = {}
    for ef_search in ef_searches:
        latencies = []
        top_k_indices = []
        distance_computations = []

        for query in queries:
            start = time.time()
            _, indices = index.search_single(
                query=query,
                K=k,
                ef_search=ef_search,
            )
            end = time.time()
            latencies.append(end - start)
            top_k_indices.append(indices)

            num_distances = index.get_query_distance_computations()
            distance_computations.append(num_distances)
        recall = compute_recall(
            queries,
            ground_truth,
            top_k_indices=top_k_indices,
            k=k,
        )
        values = {
            "recall": recall,
            "p99_latency": np.percentile(latencies, 99),
            "p90_latency": np.percentile(latencies, 90),
            "p50_latency": np.percentile(latencies, 50),
            "mean_latency": np.mean(latencies),
            "p99_distances": np.percentile(distance_computations, 99),
            "p90_distances": np.percentile(distance_computations, 90),
            "p50_distances": np.percentile(distance_computations, 50),
            "mean_distances": np.mean(distance_computations),
        }
        output[ef_search] = values
    return output


# Get your numpy-formatted dataset.
# '''
# SIFT config:
dataset = np.load("data/sift-128-euclidean/sift-128-euclidean.train.npy")
dataset_size = dataset.shape[0]
dataset_dimension = dataset.shape[1]
queries = np.load("data/sift-128-euclidean/sift-128-euclidean.test.npy")
queries = np.array(queries)
gtruth = np.load("data/sift-128-euclidean/sift-128-euclidean.gtruth.npy")
gtruth = np.array(gtruth)
# Define index construction parameters.
distance_type = "l2"
max_edges_per_node = 16
ef_construction = 30
num_build_threads = 8

ef_searches = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000]
# '''

"""
# Glove-100 config:
dataset = np.load("data/glove-200-angular/glove-200-angular.train.npy")
dataset_size = dataset.shape[0]
dataset_dimension = dataset.shape[1]
queries = np.load("data/glove-200-angular/glove-200-angular.test.npy")
queries = np.array(queries)
gtruth = np.load("data/glove-200-angular/glove-200-angular.gtruth.npy")
gtruth = np.array(gtruth)
# Define index construction parameters.
distance_type = "angular"
max_edges_per_node = 32
ef_construction = 100
num_build_threads = 8
# ef_searches = [100,200,300,400,500,600,700,800,900,1000,2000,5000,10000]
# ef_searches = list(np.arange(100, 10001, 100))
# ef_searches.extend([20000, 30000, 40000, 50000])
# The above one took wayyyy too long.
ef_searches = [100,200,350,500,1000,1500,2000,3500,5000,10000,20000,30000,50000,100000]
# """

for heuristic in ALL_HEURISTICS:
    print(f"Method: {heuristic}")
    start_time = time.time()
    # Create index configuration and pre-allocate memory
    params = BuildParameters(
        dim=dataset_dimension,
        M=max_edges_per_node,
        dataset_size=dataset_size,
        data_type=DataType.float32,
    )

    allocator = MemoryAllocator(params=params)

    index = flatnav.index.create(
        distance_type=distance_type,
        params=params,
        mem_allocator=allocator,
        verbose=True,
        collect_stats=True,
    )

    try:
        index.set_num_threads(num_build_threads)
        index.set_pruning_heuristic(heuristic=heuristic)

        # Now index the dataset
        print("Building index...")
        index.add(data=dataset, ef_construction=ef_construction)
        print("Index built.")
        index.set_num_threads(1)
        index.get_query_distance_computations()  # Reset the counter.

        print("Running queries...")
        results = compute_metrics(
            index=index,
            queries=queries,
            ground_truth=gtruth,
            ef_searches=ef_searches,
            k=100,
        )

        end_time = time.time()
        dict_name = str(heuristic).split(".")[1]
        print(f"Experiment took {end_time-start_time:.2f} seconds.")
        print(f"\n{dict_name} = {results}\n")
    except Exception as e:
        print(f"Could not run experiment {heuristic}")
        raise (e)

    del index
    del allocator