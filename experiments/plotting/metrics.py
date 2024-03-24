from typing import List
import numpy as np


def compute_recall(
    queries: np.ndarray, ground_truth: np.ndarray, top_k_indices: List[int], k: int
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


RUNTIME_METRICS = {
    "recall@k": {
        "Description": "Recall@k",
        "worst_value": float("inf"),
        "range": [0, 1],
        "function": compute_recall,
    },
    "qps": {
        "Description": "Queries per second",
        "worst_value": float("-inf"),
        "function": lambda querying_time, num_queries: num_queries / querying_time,
    },
    "p50": {
        "Description": "50th percentile latency (ms)",
        "worst_value": float("inf"),
        "function": lambda latencies: np.percentile(latencies, 50) * 1000,
    },
    "p90": {
        "Description": "90th percentile latency (ms)",
        "worst_value": float("inf"),
        "function": lambda latencies: np.percentile(latencies, 90) * 1000,
    },
    "p95": {
        "Description": "95th percentile latency (ms)",
        "worst_value": float("inf"),
        "function": lambda latencies: np.percentile(latencies, 95) * 1000,
    },
    "p99": {
        "Description": "99th percentile latency (ms)",
        "worst_value": float("inf"),
        "function": lambda latencies: np.percentile(latencies, 99) * 1000,
    },
    "p999": {
        "Description": "99.9th percentile latency (ms)",
        "worst_value": float("inf"),
        "function": lambda latencies: np.percentile(latencies, 99.9) * 1000,
    },
    "distance_computations": {
        "Description": "Average number of distance computations per query",
        "worst_value": float("inf"),
        "function": lambda distance_computations, num_queries: distance_computations / num_queries,
    },
    "index_size": {
        "Description": "Index size (bytes)",
        "worst_value": float("inf"),
    },
    "build_time": {
        "Description": "Index build time (s)",
        "worst_value": float("inf"),
    },
}
