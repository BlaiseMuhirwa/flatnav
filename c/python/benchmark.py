#!/usr/bin/env python3
"""
Benchmark script comparing FlatNav C API vs C++ API Python bindings.

This script benchmarks both implementations on the SIFT-128 dataset
and compares build time, search time, and recall.

Usage:
    python benchmark.py [options]

Options:
    --n-train N       Number of training vectors (default: 100000)
    --n-queries N     Number of queries (default: 10000)
    --k K             Number of neighbors to retrieve (default: 10)
    --ef-construction EF  Construction ef (default: 200)
    --ef-search EF    Search ef (default: 200)
    --m M             Max edges per node (default: 32)
"""

import argparse
import sys
import os
import time
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Data paths (relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "sift-128-euclidean")
TRAIN_PATH = os.path.join(DATA_DIR, "sift-128-euclidean.train.npy")
TEST_PATH = os.path.join(DATA_DIR, "sift-128-euclidean.test.npy")
GTRUTH_PATH = os.path.join(DATA_DIR, "sift-128-euclidean.gtruth.npy")


def load_data(n_train=None, n_queries=None):
    """Load SIFT-128 dataset."""
    print("Loading SIFT-128 dataset...")

    train_data = np.load(TRAIN_PATH).astype(np.float32)
    test_data = np.load(TEST_PATH).astype(np.float32)
    ground_truth = np.load(GTRUTH_PATH).astype(np.int32)

    if n_train and n_train < len(train_data):
        train_data = train_data[:n_train]

    if n_queries and n_queries < len(test_data):
        test_data = test_data[:n_queries]
        ground_truth = ground_truth[:n_queries]

    print(f"  Train: {train_data.shape}")
    print(f"  Test: {test_data.shape}")
    print(f"  Ground truth: {ground_truth.shape}")

    return train_data, test_data, ground_truth


def compute_recall(results, ground_truth, k, n_train=None):
    """Compute mean recall@k.

    If n_train is specified, filters ground truth to only include labels < n_train.
    This is needed when indexing a subset of the full dataset.
    """
    n_queries = len(results)
    total_recall = 0.0

    for i in range(n_queries):
        result_set = set(results[i][:k])

        # Filter ground truth if needed
        if n_train is not None:
            filtered_gt = [g for g in ground_truth[i] if g < n_train][:k]
        else:
            filtered_gt = ground_truth[i][:k]

        if len(filtered_gt) == 0:
            continue

        gt_set = set(filtered_gt)
        hits = len(result_set & gt_set)
        total_recall += hits / len(filtered_gt)

    return total_recall / n_queries


def benchmark_c_api(train_data, test_data, ground_truth, args):
    """Benchmark FlatNav C API."""
    print("\n" + "=" * 60)
    print("FlatNav C API Benchmark")
    print("=" * 60)

    import flatnav_c

    print(f"Version: {flatnav_c.version()}")
    print(f"Build: {flatnav_c.build_info()}")

    n_train, dim = train_data.shape
    n_test = len(test_data)

    # Create index
    print(f"\nCreating index (dim={dim}, M={args.m}, max_nodes={n_train})...")
    index = flatnav_c.Index(
        dimension=dim,
        max_nodes=n_train,
        max_edges=args.m,
        metric="l2",
        dtype="float32"
    )

    # Build index
    print(f"Building index (ef_construction={args.ef_construction}, num_init=100)...")
    build_start = time.perf_counter()

    for i, vec in enumerate(train_data):
        index.add(vec, label=i, ef_construction=args.ef_construction, num_init=100)
        if (i + 1) % 10000 == 0:
            print(f"\r  Added {i + 1}/{n_train} vectors...", end="", flush=True)

    build_time = time.perf_counter() - build_start
    print(f"\n  Build time: {build_time:.2f}s ({n_train / build_time:.0f} vectors/sec)")

    # Search
    print(f"\nSearching (k={args.k}, ef_search={args.ef_search}, num_init=100)...")
    search_start = time.perf_counter()

    all_labels = []
    all_distances = []
    for i, query in enumerate(test_data):
        labels, distances = index.search(
            query, k=args.k, ef_search=args.ef_search, num_init=100
        )
        all_labels.append(labels)
        all_distances.append(distances)

    search_time = time.perf_counter() - search_start
    qps = n_test / search_time
    print(f"  Search time: {search_time:.3f}s ({qps:.0f} QPS)")

    # Compute recall
    results = np.array(all_labels)
    recall = compute_recall(results, ground_truth, args.k, n_train=n_train)
    print(f"  Recall@{args.k}: {recall:.4f} ({recall * 100:.1f}%)")

    return {
        "build_time": build_time,
        "search_time": search_time,
        "qps": qps,
        "recall": recall
    }


def benchmark_cpp_api(train_data, test_data, ground_truth, args):
    """Benchmark FlatNav C++ API."""
    print("\n" + "=" * 60)
    print("FlatNav C++ API Benchmark")
    print("=" * 60)

    try:
        from flatnav.index import create
    except ImportError as e:
        print(f"Error: Could not import flatnav C++ bindings: {e}")
        print("Make sure to install the flatnav package from python-bindings/")
        return None

    n_train, dim = train_data.shape
    n_test = len(test_data)

    # Create index
    print(f"\nCreating index (dim={dim}, M={args.m}, max_nodes={n_train})...")
    index = create(
        distance_type="l2",
        dim=dim,
        dataset_size=n_train,
        max_edges_per_node=args.m,
        verbose=False
    )

    # Build index
    print(f"Building index (ef_construction={args.ef_construction})...")
    build_start = time.perf_counter()

    index.add(data=train_data, ef_construction=args.ef_construction)

    build_time = time.perf_counter() - build_start
    print(f"  Build time: {build_time:.2f}s ({n_train / build_time:.0f} vectors/sec)")

    # Search
    print(f"\nSearching (k={args.k}, ef_search={args.ef_search})...")
    search_start = time.perf_counter()

    distances, labels = index.search(
        queries=test_data, ef_search=args.ef_search, K=args.k
    )

    search_time = time.perf_counter() - search_start
    qps = n_test / search_time
    print(f"  Search time: {search_time:.3f}s ({qps:.0f} QPS)")

    # Compute recall
    recall = compute_recall(labels, ground_truth, args.k, n_train=n_train)
    print(f"  Recall@{args.k}: {recall:.4f} ({recall * 100:.1f}%)")

    return {
        "build_time": build_time,
        "search_time": search_time,
        "qps": qps,
        "recall": recall
    }


def print_comparison(c_results, cpp_results):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    if cpp_results is None:
        print("C++ API not available for comparison")
        return

    print(f"\n{'Metric':<20} {'C API':>15} {'C++ API':>15} {'Ratio':>10}")
    print("-" * 60)

    for metric, c_val in c_results.items():
        cpp_val = cpp_results[metric]
        if metric in ["build_time", "search_time"]:
            ratio = c_val / cpp_val if cpp_val > 0 else float('inf')
            print(f"{metric:<20} {c_val:>14.2f}s {cpp_val:>14.2f}s {ratio:>9.2f}x")
        elif metric == "qps":
            ratio = c_val / cpp_val if cpp_val > 0 else float('inf')
            print(f"{metric:<20} {c_val:>15.0f} {cpp_val:>15.0f} {ratio:>9.2f}x")
        elif metric == "recall":
            print(f"{metric:<20} {c_val:>14.4f} {cpp_val:>14.4f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark FlatNav C vs C++ API")
    parser.add_argument("--n-train", type=int, default=100000,
                        help="Number of training vectors")
    parser.add_argument("--n-queries", type=int, default=10000,
                        help="Number of queries")
    parser.add_argument("--k", type=int, default=10,
                        help="Number of neighbors to retrieve")
    parser.add_argument("--ef-construction", type=int, default=200,
                        help="Construction ef parameter")
    parser.add_argument("--ef-search", type=int, default=200,
                        help="Search ef parameter")
    parser.add_argument("--m", type=int, default=32,
                        help="Max edges per node (M)")
    parser.add_argument("--c-only", action="store_true",
                        help="Only benchmark C API")
    parser.add_argument("--cpp-only", action="store_true",
                        help="Only benchmark C++ API")

    args = parser.parse_args()

    # Check data exists
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Dataset not found at {DATA_DIR}")
        print("Please download the SIFT-128 dataset first.")
        return 1

    # Load data
    train_data, test_data, ground_truth = load_data(args.n_train, args.n_queries)

    c_results = None
    cpp_results = None

    # Run benchmarks
    if not args.cpp_only:
        c_results = benchmark_c_api(train_data, test_data, ground_truth, args)

    if not args.c_only:
        cpp_results = benchmark_cpp_api(train_data, test_data, ground_truth, args)

    # Print comparison
    if c_results and cpp_results:
        print_comparison(c_results, cpp_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
