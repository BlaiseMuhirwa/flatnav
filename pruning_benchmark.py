import numpy as np
import flatnav
from flatnav.data_type import DataType 
import time


def compute_recall(
    queries, ground_truth, top_k_indices, k
) -> float:
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
            "recall" : recall,
            "p99_latency" : np.percentile(latencies, 99),
            "p90_latency" : np.percentile(latencies, 90),
            "p50_latency" : np.percentile(latencies, 50),
            "mean_latency" : np.mean(latencies),
            "p99_distances" : np.percentile(distance_computations, 99),
            "p90_distances" : np.percentile(distance_computations, 90),
            "p50_distances" : np.percentile(distance_computations, 50),
            "mean_distances" : np.mean(distance_computations),
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
ef_construction = 100
num_build_threads = 8

ef_searches = [100,200,300,400,500,600,700,800,900,1000,2000,5000]
# '''

'''
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
# '''

pruning_methods = {
    0: "Arya-Mount",
    1: "Arya-Mount-Reproduction",
    13: "Cheap-Outdegree-Conditional-M-Over-4",
    28: "Cheap-Outdegree-Conditional-2",
    29: "Cheap-Outdegree-Conditional-4",
    30: "Cheap-Outdegree-Conditional-6",
    31: "Cheap-Outdegree-Conditional-8",
    32: "Cheap-Outdegree-Conditional-10",
    33: "Cheap-Outdegree-Conditional-12",
    34: "Cheap-Outdegree-Conditional-16",
    27: "Arya-Mount-DiskANN-Inverse",
    20: "Arya-Mount-Random-On-Rejects-1p",
    21: "Arya-Mount-Random-On-Rejects-5p",
    22: "Arya-Mount-Random-On-Rejects-10p",
    23: "Arya-Mount-Sigmoid-On-Rejects-0p1",
    24: "Arya-Mount-Sigmoid-On-Rejects-1",
    25: "Arya-Mount-Sigmoid-On-Rejects-5",
    26: "Arya-Mount-Sigmoid-On-Rejects-10",
    12: "Neighborhood-Overlap",
    16: "Sigmoid-Ratio-1",
    17: "Sigmoid-Ratio-5",
    18: "Sigmoid-Ratio-10",
    2: "Arya-Mount-DiskANN",
    3: "Nearest-M",
    4: "Furthest-M",
    5: "Median-Adaptive",
    6: "Top-M-Mean-Adaptive",
    7: "Mean-Baseline-Dist",
    8: "Quantile-Not-Min-20p",
    9: "Quantile-Not-Min-10p",
    10: "Arya-Mount-Reversed",
    11: "Probabilistic-Rank",
    14: "Large-Outdegree-Conditional",
    15: "Geometric-Mean",
    19: "Arya-Mount-Shuffled",
    35: "One-Hop-Spanner",
    36: "Arya-Mount-Plus-Spanner",
    37: "Cheap-Outdegree-Conditional-M", # Sanity check - should be as bad as Nearest-M
}

for algorithm_id, algorithm_name in pruning_methods.items():
    print(f"Method: {algorithm_name}")
    start_time = time.time()
    # Create index configuration and pre-allocate memory
    index = flatnav.index.create(
        distance_type=distance_type,
        index_data_type=DataType.float32,
        dim=dataset_dimension,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=True,
    )
    try:
        index.set_num_threads(num_build_threads)
        index.set_pruning_algorithm(algorithm_id)

        # Now index the dataset 
        index.add(data=dataset, ef_construction=ef_construction)
        index.set_num_threads(1)
        index.get_query_distance_computations()  # Reset the counter.

        results = compute_metrics(
            index = index,
            queries = queries,
            ground_truth = gtruth,
            ef_searches = ef_searches,
            k=100,
        )

        end_time = time.time()
        dict_name = algorithm_name.replace("-","_").lower()
        print(f"Experiment took {end_time-start_time:.2f} seconds.")
        print(f"\n{dict_name} = {results}\n")
    except Exception as e:
        print(f"Could not run experiment {algorithm_name}")
        raise(e)

    del index
