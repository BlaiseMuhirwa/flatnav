import platform, socket, psutil 
import os 
import numpy as np 

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

def compute_recall(index, queries: np.ndarray, ground_truth: np.ndarray, k=100):
    """
    Compute recall for given queries, ground truth, and an index.

    Args:
        - index: The Faiss index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.

    Returns:
        Mean recall over all queries.
    """
    _, top_k_indices = index.search(queries, k)

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)
    return recall