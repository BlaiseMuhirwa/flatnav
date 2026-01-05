#!/usr/bin/env python3
"""
Evaluate two-pass construction strategies on large-scale benchmarks.

This script compares:
- Baseline: Single-pass construction with M=8
- Two-pass strategies: M=4 + M=4 with different optimization strategies

Hypothesis: Two-pass with M=4+M=4 achieves similar recall to single-pass M=8
but with faster construction time due to cheaper graph maintenance.

Usage:
    python evaluate_two_pass.py \
        --dataset /path/to/train.npy \
        --queries /path/to/queries.npy \
        --gtruth /path/to/ground_truth.npy \
        --strategies hubness edge_quality insertion_order \
        --output results.json
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

import flatnav
from flatnav import index as flatnav_index
from flatnav import TwoPassStrategy, Pass2CandidateMethod

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    strategy: str
    construction_time_seconds: float
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    qps_at_90_recall: float
    qps_at_95_recall: float
    M_total: int
    ef_construction_total: int
    num_vectors: int
    dimension: int


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    # Baseline parameters
    M_baseline: int = 16
    ef_construction_baseline: int = 100

    # Two-pass parameters
    M_pass1: int = 8
    M_pass2: int = 8
    ef_construction_pass1: int = 25
    ef_construction_pass2: int = 100

    # Common parameters
    num_initializations: int = 100
    num_threads: int = 8

    # Strategy-specific parameters
    hubness_penalty_weight: float = 0.1
    edge_quality_threshold: float = 0.5

    # Pass 2 candidate method parameters
    pass2_candidate_method: str = "beam_search"  # "beam_search" or "neighbor_expansion"
    neighbor_expansion_hops: int = 2  # 2 or 3

    # Search parameters
    ef_search_values: List[int] = None
    K: int = 100

    def __post_init__(self):
        if self.ef_search_values is None:
            self.ef_search_values = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000]


def compute_recall(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Compute recall@k.

    Args:
        predictions: Shape (num_queries, num_predictions) - predicted neighbor IDs
        ground_truth: Shape (num_queries, num_gt) - ground truth neighbor IDs
        k: Number of neighbors to consider

    Returns:
        Recall@k as a float between 0 and 1
    """
    num_queries = predictions.shape[0]

    # Limit to k predictions and k ground truth
    pred_k = predictions[:, :k] if predictions.shape[1] >= k else predictions
    gt_k = ground_truth[:, :k] if ground_truth.shape[1] >= k else ground_truth

    total_correct = 0
    for i in range(num_queries):
        pred_set = set(pred_k[i].tolist())
        gt_set = set(gt_k[i].tolist())
        total_correct += len(pred_set & gt_set)

    return total_correct / (num_queries * k)


def find_ef_for_target_recall(
    index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    target_recall: float,
    K: int,
    ef_values: List[int],
    num_initializations: int = 100
) -> tuple:
    """
    Find the ef_search value that achieves target recall and return QPS.

    Returns:
        (ef_search, recall, qps) tuple
    """
    best_ef = None
    best_recall = 0.0
    best_qps = 0.0

    for ef in sorted(ef_values):
        start = time.time()
        _, predictions = index.search(queries, K, ef, num_initializations)
        elapsed = time.time() - start

        recall = compute_recall(predictions, ground_truth, K)
        qps = len(queries) / elapsed

        if recall >= target_recall and (best_ef is None or qps > best_qps):
            best_ef = ef
            best_recall = recall
            best_qps = qps

        if best_recall >= target_recall:
            break

    return best_ef, best_recall, best_qps


def build_baseline_index(
    data: np.ndarray,
    config: ExperimentConfig,
    distance_type: str = "l2"
) -> Any:
    """Build a baseline single-pass index."""
    num_vectors, dim = data.shape

    index = flatnav_index.create(
        distance_type=distance_type,
        dim=dim,
        dataset_size=num_vectors,
        max_edges_per_node=config.M_baseline
    )
    index.set_num_threads(config.num_threads)
    index.add(data, config.ef_construction_baseline, config.num_initializations)

    return index


def build_two_pass_index(
    data: np.ndarray,
    config: ExperimentConfig,
    strategy: str,
    distance_type: str = "l2"
) -> Any:
    """Build a two-pass index with the specified strategy."""
    num_vectors, dim = data.shape

    strategy_map = {
        "hubness": TwoPassStrategy.HUBNESS_SCORING,
        "edge_quality": TwoPassStrategy.EDGE_QUALITY_SCORING,
        "insertion_order": TwoPassStrategy.INSERTION_ORDER_OPT,
    }

    strategy_enum = strategy_map.get(strategy)
    if strategy_enum is None:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Map string to enum for pass2_candidate_method
    candidate_method_map = {
        "beam_search": Pass2CandidateMethod.BEAM_SEARCH,
        "neighbor_expansion": Pass2CandidateMethod.NEIGHBOR_EXPANSION,
    }
    candidate_method = candidate_method_map.get(
        config.pass2_candidate_method,
        Pass2CandidateMethod.BEAM_SEARCH
    )

    index = flatnav_index.create_two_pass(
        distance_type=distance_type,
        dim=dim,
        dataset_size=num_vectors,
        data=data,
        strategy=strategy_enum,
        M_pass1=config.M_pass1,
        M_pass2=config.M_pass2,
        ef_construction_pass1=config.ef_construction_pass1,
        ef_construction_pass2=config.ef_construction_pass2,
        num_initializations=config.num_initializations,
        hubness_penalty_weight=config.hubness_penalty_weight,
        edge_quality_threshold=config.edge_quality_threshold,
        num_threads=config.num_threads,
        pass2_candidate_method=candidate_method,
        neighbor_expansion_hops=config.neighbor_expansion_hops,
    )

    return index


def evaluate_index(
    index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    config: ExperimentConfig
) -> Dict[str, float]:
    """Evaluate an index on the given queries."""
    results = {}

    # Compute recall at different K values
    for k in [1, 10, 100]:
        if k > ground_truth.shape[1]:
            continue
        ef = max(config.ef_search_values)
        _, predictions = index.search(queries, k, ef, config.num_initializations)
        results[f"recall_at_{k}"] = compute_recall(predictions, ground_truth, k)

    # Find QPS at target recalls
    for target_recall in [0.80, 0.90, 0.95]:
        ef, recall, qps = find_ef_for_target_recall(
            index, queries, ground_truth, target_recall,
            config.K, config.ef_search_values, config.num_initializations
        )
        results[f"qps_at_{int(target_recall*100)}_recall"] = qps

    return results


def run_benchmark(
    data: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    strategy: str,
    config: ExperimentConfig,
    distance_type: str = "l2"
) -> BenchmarkResult:
    """Run a single benchmark for a given strategy."""
    num_vectors, dim = data.shape

    print(f"  Building index with strategy: {strategy}...")

    start_time = time.time()
    if strategy == "baseline":
        index = build_baseline_index(data, config, distance_type)
        M_total = config.M_baseline
        ef_total = config.ef_construction_baseline
    else:
        index = build_two_pass_index(data, config, strategy, distance_type)
        M_total = config.M_pass1 + config.M_pass2
        ef_total = config.ef_construction_pass1 + config.ef_construction_pass2
    construction_time = time.time() - start_time

    print(f"    Construction time: {construction_time:.2f}s")

    print(f"  Evaluating index...")
    eval_results = evaluate_index(index, queries, ground_truth, config)

    return BenchmarkResult(
        strategy=strategy,
        construction_time_seconds=construction_time,
        recall_at_1=eval_results.get("recall_at_1", 0.0),
        recall_at_10=eval_results.get("recall_at_10", 0.0),
        recall_at_100=eval_results.get("recall_at_100", 0.0),
        qps_at_90_recall=eval_results.get("qps_at_90_recall", 0.0),
        qps_at_95_recall=eval_results.get("qps_at_95_recall", 0.0),
        M_total=M_total,
        ef_construction_total=ef_total,
        num_vectors=num_vectors,
        dimension=dim
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two-pass construction strategies"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to training data (.npy file)"
    )
    parser.add_argument(
        "--queries", type=str, required=True,
        help="Path to query data (.npy file)"
    )
    parser.add_argument(
        "--gtruth", type=str, required=True,
        help="Path to ground truth (.npy file)"
    )
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=["baseline", "hubness", "edge_quality", "insertion_order"],
        help="Strategies to evaluate"
    )
    parser.add_argument(
        "--distance-type", type=str, default="l2",
        choices=["l2", "angular"],
        help="Distance metric to use"
    )
    parser.add_argument(
        "--output", type=str, default="two_pass_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--num-threads", type=int, default=1,
        help="Number of threads for construction"
    )
    parser.add_argument(
        "--M-baseline", type=int, default=8,
        help="M value for baseline single-pass"
    )
    parser.add_argument(
        "--M-pass1", type=int, default=4,
        help="M value for two-pass Pass 1"
    )
    parser.add_argument(
        "--M-pass2", type=int, default=4,
        help="M value for two-pass Pass 2"
    )
    parser.add_argument(
        "--ef-construction-baseline", type=int, default=100,
        help="ef_construction for baseline"
    )
    parser.add_argument(
        "--ef-construction-pass1", type=int, default=80,
        help="ef_construction for two-pass Pass 1"
    )
    parser.add_argument(
        "--ef-construction-pass2", type=int, default=25,
        help="ef_construction for two-pass Pass 2"
    )
    parser.add_argument(
        "--pass2-candidate-method", type=str, default="beam_search",
        choices=["beam_search", "neighbor_expansion"],
        help="Method for finding Pass 2 candidates (default: beam_search)"
    )
    parser.add_argument(
        "--neighbor-expansion-hops", type=int, default=2,
        help="Number of hops for neighbor expansion (2 or 3, default: 2)"
    )
    parser.add_argument(
        "--hubness-penalty-weight", type=float, default=0.1,
        help="Weight for hubness penalty in HUBNESS_SCORING strategy (default: 0.1)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading dataset from {args.dataset}...")
    data = np.load(args.dataset)
    print(f"  Shape: {data.shape}")

    print(f"Loading queries from {args.queries}...")
    queries = np.load(args.queries)
    print(f"  Shape: {queries.shape}")

    print(f"Loading ground truth from {args.gtruth}...")
    ground_truth = np.load(args.gtruth)
    print(f"  Shape: {ground_truth.shape}")

    # Create config
    config = ExperimentConfig(
        M_baseline=args.M_baseline,
        ef_construction_baseline=args.ef_construction_baseline,
        M_pass1=args.M_pass1,
        M_pass2=args.M_pass2,
        ef_construction_pass1=args.ef_construction_pass1,
        ef_construction_pass2=args.ef_construction_pass2,
        num_threads=args.num_threads,
        pass2_candidate_method=args.pass2_candidate_method,
        neighbor_expansion_hops=args.neighbor_expansion_hops,
        hubness_penalty_weight=args.hubness_penalty_weight,
    )

    # Run benchmarks
    results = []
    for strategy in args.strategies:
        print(f"\nRunning benchmark for strategy: {strategy}")

        result = run_benchmark(
            data, queries, ground_truth, strategy, config, args.distance_type
        )

        results.append(asdict(result))
        print(f"  Recall@10: {result.recall_at_10:.4f}")
        print(f"  Construction time: {result.construction_time_seconds:.2f}s")

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Strategy':<20} {'Time (s)':<12} {'Recall@1':<12} {'Recall@10':<12} {'Recall@100':<12} {'Speedup':<12}")
    print("-" * 100)

    baseline_time = None
    for r in results:
        if r["strategy"] == "baseline":
            baseline_time = r["construction_time_seconds"]
            break

    for r in results:
        speedup = baseline_time / r["construction_time_seconds"] if baseline_time else 1.0
        print(f"{r['strategy']:<20} {r['construction_time_seconds']:<12.2f} "
              f"{r['recall_at_1']:<12.4f} {r['recall_at_10']:<12.4f} "
              f"{r['recall_at_100']:<12.4f} {speedup:<12.2f}x")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "config": asdict(config),
            "results": results
        }, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
