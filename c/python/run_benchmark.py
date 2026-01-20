#!/usr/bin/env python3
"""
End-to-end benchmark script for FlatNav C API.

This script benchmarks the C API on various datasets and sweeps across
different ef_construction and ef_search parameters.

Usage:
    python run_benchmark.py \
        --dataset-name sift-128 \
        --dataset /path/to/train.npy \
        --queries /path/to/test.npy \
        --gtruth /path/to/gtruth.npy \
        --metric l2 \
        --num-node-links 32 \
        --ef-construction 50 100 200 \
        --ef-search 100 200 300 500 \
        --num-build-threads 8 \
        --num-search-threads 1
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Add the package to path
sys.path.insert(0, str(Path(__file__).parent))
import flatnav_c


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    dataset_name: str
    n_vectors: int
    dimension: int
    M: int
    ef_construction: int
    ef_search: int
    num_build_threads: int
    num_search_threads: int
    build_time_sec: float
    search_time_sec: float
    vectors_per_sec: float
    qps: float
    recall_at_k: float
    k: int


def load_npy(path: str) -> np.ndarray:
    """Load numpy file."""
    arr = np.load(path)
    return arr


def compute_recall(results: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute mean recall@k."""
    n_queries = len(results)
    total_recall = 0.0

    for i in range(n_queries):
        result_set = set(results[i][:k].tolist())
        gt_set = set(ground_truth[i][:k].tolist())
        hits = len(result_set & gt_set)
        total_recall += hits / min(k, len(gt_set))

    return total_recall / n_queries


def add_vectors_parallel(
    index: flatnav_c.Index,
    vectors: np.ndarray,
    ef_construction: int,
    num_init: int,
    num_threads: int,
) -> float:
    """Add vectors to index using multiple threads. Returns build time in seconds."""
    n = len(vectors)

    if num_threads == 1:
        # Single-threaded: add sequentially
        start = time.perf_counter()
        for i in range(n):
            index.add(vectors[i], label=i, ef_construction=ef_construction, num_init=num_init)
            if (i + 1) % 100000 == 0:
                elapsed = time.perf_counter() - start
                print(f"    Added {i + 1}/{n} vectors ({(i + 1) / elapsed:.0f} vec/sec)...")
        return time.perf_counter() - start

    # Multi-threaded: use thread pool
    # Note: The C API is thread-safe for adds
    from threading import Lock
    import threading

    counter = [0]
    counter_lock = Lock()
    start = time.perf_counter()

    def add_range(start_idx: int, end_idx: int):
        for i in range(start_idx, end_idx):
            index.add(vectors[i], label=i, ef_construction=ef_construction, num_init=num_init)
            with counter_lock:
                counter[0] += 1
                if counter[0] % 100000 == 0:
                    elapsed = time.perf_counter() - start
                    print(f"    Added {counter[0]}/{n} vectors ({counter[0] / elapsed:.0f} vec/sec)...")

    # Split work across threads
    chunk_size = (n + num_threads - 1) // num_threads
    threads = []
    for t in range(num_threads):
        start_idx = t * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        if start_idx >= n:
            break
        thread = threading.Thread(target=add_range, args=(start_idx, end_idx))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return time.perf_counter() - start


def search_parallel(
    index: flatnav_c.Index,
    queries: np.ndarray,
    k: int,
    ef_search: int,
    num_init: int,
    num_threads: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Search index using multiple threads. Returns (labels, distances, time)."""
    n = len(queries)
    all_labels = np.zeros((n, k), dtype=np.int32)
    all_distances = np.zeros((n, k), dtype=np.float32)

    if num_threads == 1:
        start = time.perf_counter()
        for i in range(n):
            labels, distances = index.search(queries[i], k=k, ef_search=ef_search, num_init=num_init)
            all_labels[i] = labels
            all_distances[i] = distances
        search_time = time.perf_counter() - start
        return all_labels, all_distances, search_time

    # Multi-threaded search
    import threading
    from threading import Lock

    start = time.perf_counter()

    def search_range(start_idx: int, end_idx: int):
        for i in range(start_idx, end_idx):
            labels, distances = index.search(queries[i], k=k, ef_search=ef_search, num_init=num_init)
            all_labels[i] = labels
            all_distances[i] = distances

    chunk_size = (n + num_threads - 1) // num_threads
    threads = []
    for t in range(num_threads):
        start_idx = t * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        if start_idx >= n:
            break
        thread = threading.Thread(target=search_range, args=(start_idx, end_idx))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    search_time = time.perf_counter() - start
    return all_labels, all_distances, search_time


def run_benchmark(
    dataset_name: str,
    train_data: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    M: int,
    ef_construction: int,
    ef_search_values: List[int],
    k: int,
    metric: str,
    num_build_threads: int,
    num_search_threads: int,
    num_init: int = 100,
) -> List[BenchmarkResult]:
    """Run benchmark for a single ef_construction value, sweeping ef_search."""
    n_vectors, dim = train_data.shape
    n_queries = len(queries)

    print(f"\n{'=' * 70}")
    print(f"Building index: M={M}, ef_construction={ef_construction}, threads={num_build_threads}")
    print(f"{'=' * 70}")

    # Create index
    index = flatnav_c.Index(
        dimension=dim,
        max_nodes=n_vectors,
        max_edges=M,
        metric=metric,
        dtype="float32",
    )

    # Build index
    build_time = add_vectors_parallel(
        index, train_data, ef_construction, num_init, num_build_threads
    )
    vectors_per_sec = n_vectors / build_time

    print(f"  Build complete: {build_time:.2f}s ({vectors_per_sec:.0f} vectors/sec)")

    # Search with different ef_search values
    results = []
    for ef_search in ef_search_values:
        print(f"\n  Searching with ef_search={ef_search}...")

        labels, distances, search_time = search_parallel(
            index, queries, k, ef_search, num_init, num_search_threads
        )

        qps = n_queries / search_time
        recall = compute_recall(labels, ground_truth, k)

        print(f"    Time: {search_time:.3f}s, QPS: {qps:.0f}, Recall@{k}: {recall:.4f}")

        results.append(BenchmarkResult(
            dataset_name=dataset_name,
            n_vectors=n_vectors,
            dimension=dim,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            num_build_threads=num_build_threads,
            num_search_threads=num_search_threads,
            build_time_sec=build_time,
            search_time_sec=search_time,
            vectors_per_sec=vectors_per_sec,
            qps=qps,
            recall_at_k=recall,
            k=k,
        ))

    return results


def main():
    parser = argparse.ArgumentParser(description="FlatNav C API Benchmark")
    parser.add_argument("--dataset-name", required=True, help="Name for the dataset")
    parser.add_argument("--dataset", required=True, help="Path to training data (npy)")
    parser.add_argument("--queries", required=True, help="Path to query data (npy)")
    parser.add_argument("--gtruth", required=True, help="Path to ground truth (npy)")
    parser.add_argument("--metric", default="l2", choices=["l2", "ip", "angular"],
                        help="Distance metric")
    parser.add_argument("--num-node-links", type=int, default=32, help="M parameter")
    parser.add_argument("--ef-construction", type=int, nargs="+", default=[100],
                        help="ef_construction values to test")
    parser.add_argument("--ef-search", type=int, nargs="+", default=[100, 200, 300],
                        help="ef_search values to test")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--num-build-threads", type=int, default=1,
                        help="Threads for building index")
    parser.add_argument("--num-search-threads", type=int, default=1,
                        help="Threads for searching")
    parser.add_argument("--num-init", type=int, default=100,
                        help="Number of random initializations")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--max-vectors", type=int, default=None,
                        help="Limit number of training vectors")

    args = parser.parse_args()

    # Print configuration
    print("FlatNav C API Benchmark")
    print("=" * 70)
    print(f"Version: {flatnav_c.version()}")
    print(f"Build: {flatnav_c.build_info()}")
    print()
    print(f"Dataset: {args.dataset_name}")
    print(f"Metric: {args.metric}")
    print(f"M: {args.num_node_links}")
    print(f"ef_construction: {args.ef_construction}")
    print(f"ef_search: {args.ef_search}")
    print(f"k: {args.k}")
    print(f"Build threads: {args.num_build_threads}")
    print(f"Search threads: {args.num_search_threads}")
    print()

    # Load data
    print("Loading data...")
    train_data = load_npy(args.dataset).astype(np.float32)
    queries = load_npy(args.queries).astype(np.float32)
    ground_truth = load_npy(args.gtruth).astype(np.int32)

    if args.max_vectors and args.max_vectors < len(train_data):
        train_data = train_data[:args.max_vectors]
        print(f"  Limited to {args.max_vectors} training vectors")

    print(f"  Train: {train_data.shape}")
    print(f"  Queries: {queries.shape}")
    print(f"  Ground truth: {ground_truth.shape}")

    # Handle metric mapping
    metric = args.metric
    if metric == "angular":
        metric = "ip"

    # Run benchmarks
    all_results = []
    for ef_construction in args.ef_construction:
        results = run_benchmark(
            dataset_name=args.dataset_name,
            train_data=train_data,
            queries=queries,
            ground_truth=ground_truth,
            M=args.num_node_links,
            ef_construction=ef_construction,
            ef_search_values=args.ef_search,
            k=args.k,
            metric=metric,
            num_build_threads=args.num_build_threads,
            num_search_threads=args.num_search_threads,
            num_init=args.num_init,
        )
        all_results.extend(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'ef_c':>6} {'ef_s':>6} {'Build(s)':>10} {'Vec/s':>10} {'QPS':>10} {'Recall':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r.ef_construction:>6} {r.ef_search:>6} {r.build_time_sec:>10.2f} "
              f"{r.vectors_per_sec:>10.0f} {r.qps:>10.0f} {r.recall_at_k:>10.4f}")

    # Save results
    if args.output:
        output_data = {
            "config": {
                "dataset_name": args.dataset_name,
                "n_vectors": train_data.shape[0],
                "dimension": train_data.shape[1],
                "n_queries": len(queries),
                "metric": args.metric,
                "M": args.num_node_links,
                "k": args.k,
                "num_build_threads": args.num_build_threads,
                "num_search_threads": args.num_search_threads,
                "flatnav_version": flatnav_c.version(),
                "build_info": flatnav_c.build_info(),
            },
            "results": [
                {
                    "ef_construction": r.ef_construction,
                    "ef_search": r.ef_search,
                    "build_time_sec": r.build_time_sec,
                    "vectors_per_sec": r.vectors_per_sec,
                    "search_time_sec": r.search_time_sec,
                    "qps": r.qps,
                    "recall_at_k": r.recall_at_k,
                }
                for r in all_results
            ],
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
