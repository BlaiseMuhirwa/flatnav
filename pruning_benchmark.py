import numpy as np
import flatnav
from flatnav.data_type import DataType 
import time
import json
import os

def compute_recall(queries, ground_truth, top_k_indices, k) -> float:
    ground_truth_sets = [set(gt) for gt in ground_truth]
    mean_recall = 0
    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx])
        mean_recall += query_recall / k
    return mean_recall / len(queries)


def compute_metrics(index, queries, ground_truth, ef_searches, k=100):
    output = {}
    for ef_search in ef_searches:
        latencies = []
        top_k_indices = []
        distance_computations = []
        for query in queries:
            start = time.time()
            _, indices = index.search_single(query=query, K=k, ef_search=ef_search)
            end = time.time()
            latencies.append(end - start)
            top_k_indices.append(indices)
            num_distances = index.get_query_distance_computations()
            distance_computations.append(num_distances)
        recall = compute_recall(queries, ground_truth, top_k_indices=top_k_indices, k=k)
        output[ef_search] = {
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
    return output


benchmark_configs = {
    # "sift-128-euclidean": {
    #     "path_prefix": "data/sift-128-euclidean/sift-128-euclidean",
    #     "distance_type": "l2",
    #     "max_edges_per_node": 8,
    #     "ef_construction": 50,
    #     "num_build_threads": 16,
    #     "ef_searches": [100,200,300,400,500,600,700,800,900,1000],
    # },
    # "glove-25-angular": {
    #     "path_prefix": "data/glove-25-angular/glove-25-angular",
    #     "distance_type": "angular",
    #     "max_edges_per_node": 8,
    #     "ef_construction": 50,
    #     "num_build_threads": 16,
    #     "ef_searches": [100,200,300,400,500,600,700,800,900,1000],
    # },
 
    # "glove-200-angular": {
    #     "path_prefix": "data/glove-200-angular/glove-200-angular",
    #     "distance_type": "angular",
    #     "max_edges_per_node": 8,
    #     "ef_construction": 50,
    #     "num_build_threads": 16,
    #     "ef_searches": [100,200,300,400,500,600,700,800,900,1000],
    # },
    "gist-960-euclidean": {
        "path_prefix": "data/gist-960-euclidean/gist-960-euclidean",
        "distance_type": "l2",
        "max_edges_per_node": 8,
        "ef_construction": 50,
        "num_build_threads": 16,
        "ef_searches": [100,200,300,400,500,600,700,800,900,1000],
    },
    "glove-100-angular": {
        "path_prefix": "data/glove-100-angular/glove-100-angular",
        "distance_type": "angular",
        "max_edges_per_node": 8,
        "ef_construction": 50,
        "num_build_threads": 16,
        "ef_searches": [100,200,300,400,500,600,700,800,900,1000],
    }, 
}

pruning_methods = {
    0: "Arya-Mount",
    1: "Arya-Mount-Reproduction",
    13: "Cheap-Outdegree-Conditional-M-Over-4",
    28: "Cheap-Outdegree-Conditional-2",
    29: "Cheap-Outdegree-Conditional-4",
    30: "Cheap-Outdegree-Conditional-6",
    # 31: "Cheap-Outdegree-Conditional-8",
    # 32: "Cheap-Outdegree-Conditional-10",
    # 33: "Cheap-Outdegree-Conditional-12",
    # 34: "Cheap-Outdegree-Conditional-16",
    27: "Arya-Mount-DiskANN-Inverse",
    20: "Arya-Mount-Random-On-Rejects-1p",
    # 21: "Arya-Mount-Random-On-Rejects-5p",
    # 22: "Arya-Mount-Random-On-Rejects-10p",
    23: "Arya-Mount-Sigmoid-On-Rejects-0p1",
    24: "Arya-Mount-Sigmoid-On-Rejects-1",
    # 25: "Arya-Mount-Sigmoid-On-Rejects-5",
    # 26: "Arya-Mount-Sigmoid-On-Rejects-10",
    12: "Neighborhood-Overlap",
    16: "Sigmoid-Ratio-1",
    # 17: "Sigmoid-Ratio-5",
    # 18: "Sigmoid-Ratio-10",
    2: "Arya-Mount-DiskANN",
    3: "Nearest-M",
    4: "Furthest-M",
    5: "Median-Adaptive",
    # 6: "Top-M-Mean-Adaptive",
    # 7: "Mean-Baseline-Dist",
    8: "Quantile-Not-Min-20p",
    # 9: "Quantile-Not-Min-10p",
    # 10: "Arya-Mount-Reversed",
    11: "Probabilistic-Rank",
    # 14: "Large-Outdegree-Conditional",
    # 15: "Geometric-Mean",
    19: "Arya-Mount-Shuffled",
    35: "One-Hop-Spanner",
    # 36: "Arya-Mount-Plus-Spanner",
    # 37: "Cheap-Outdegree-Conditional-M",  # Sanity check - should be as bad as Nearest-M
}

os.makedirs("results-second", exist_ok=True)

for dataset_name, cfg in benchmark_configs.items():
    print(f"\n===== Running benchmarks on dataset: {dataset_name} =====")
    dataset = np.load(f"{cfg['path_prefix']}.train.npy")
    queries = np.load(f"{cfg['path_prefix']}.test.npy")
    gtruth = np.load(f"{cfg['path_prefix']}.gtruth.npy")
    dataset_size = dataset.shape[0]
    dataset_dimension = dataset.shape[1]

    result_path = f"results/{dataset_name}.json"

    # Load existing results for this dataset, or initialize
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            dataset_results = json.load(f)
    else:
        dataset_results = {}

    for algorithm_id, algorithm_name in pruning_methods.items():
        method_key = algorithm_name.replace("-", "_").lower()

        if method_key in dataset_results:
            print(f"Skipping {method_key} — already completed.")
            continue

        print(f"\nMethod: {algorithm_name}")
        start_time = time.time()

        index = flatnav.index.create(
            distance_type=cfg["distance_type"],
            index_data_type=DataType.float32,
            dim=dataset_dimension,
            dataset_size=dataset_size,
            max_edges_per_node=cfg["max_edges_per_node"],
            verbose=True,
            collect_stats=True,
        )

        try:
            index.set_num_threads(cfg["num_build_threads"])
            index.set_pruning_algorithm(algorithm_id)
            index.add(data=dataset, ef_construction=cfg["ef_construction"])
            index.set_num_threads(1)
            index.get_query_distance_computations()

            results = compute_metrics(
                index=index,
                queries=queries,
                ground_truth=gtruth,
                ef_searches=cfg["ef_searches"],
                k=100,
            )

            end_time = time.time()
            dataset_results[method_key] = results
            print(f"Experiment took {end_time - start_time:.2f} seconds.")

            # Save incrementally
            with open(result_path, "w") as f:
                json.dump(dataset_results, f, indent=2)
            print(f"✔ Saved {method_key} results to {result_path}")

        except Exception as e:
            print(f"❌ Could not run experiment {algorithm_name} on {dataset_name}")
            raise e

        del index