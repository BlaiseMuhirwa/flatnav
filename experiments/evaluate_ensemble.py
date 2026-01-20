#!/usr/bin/env python3
"""
Evaluate ensemble construction strategies on large-scale benchmarks.

This script compares:
- Baseline: Single-pass construction with M=16
- Two-pass: M=8 + M=8 with RE_PRUNE_FULL strategy
- Ensemble: k graphs with M'=8, pruned to M=16

Hypothesis: Ensemble with 2-3 cheap graphs achieves similar or better recall
to baseline single-pass with faster construction, because:
1. Different insertion orders produce different local optima
2. Unioning candidates from multiple builds covers failure cases
3. Final re-ranking from k*M' candidates approximates true optimal M

Usage:
    python evaluate_ensemble.py \
        --dataset /path/to/train.npy \
        --queries /path/to/queries.npy \
        --gtruth /path/to/ground_truth.npy \
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
from flatnav import TwoPassStrategy, Pass2CandidateMethod, EnsembleVariant


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    variant: str
    construction_time_seconds: float
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    qps_at_90_recall: float
    qps_at_95_recall: float
    M_final: int
    num_graphs: int
    ef_construction_total: int
    num_vectors: int
    dimension: int


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""
    # Baseline parameters
    M_baseline: int = 16
    ef_construction_baseline: int = 100

    # Two-pass parameters (for comparison)
    M_pass1: int = 8
    M_pass2: int = 8
    ef_construction_pass1: int = 25
    ef_construction_pass2: int = 100

    # Ensemble parameters
    num_graphs: int = 2
    M_per_graph: int = 8
    M_final: int = 16
    ef_construction_per_graph: int = 25
    ef_construction_final: int = 100

    # Common parameters
    num_initializations: int = 100
    num_threads: int = 8
    hubness_penalty_weight: float = 0.1
    seed_base: int = 42

    # Search parameters
    ef_search_values: List[int] = None
    K: int = 100

    def __post_init__(self):
        if self.ef_search_values is None:
            self.ef_search_values = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000]


def compute_recall(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    num_queries = predictions.shape[0]
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
    """Find the ef_search value that achieves target recall and return QPS."""
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
    distance_type: str = "l2"
) -> Any:
    """Build a two-pass index with RE_PRUNE_FULL strategy."""
    num_vectors, dim = data.shape

    index = flatnav_index.create_two_pass(
        distance_type=distance_type,
        dim=dim,
        dataset_size=num_vectors,
        data=data,
        strategy=TwoPassStrategy.RE_PRUNE_FULL,
        M_pass1=config.M_pass1,
        M_pass2=config.M_pass2,
        ef_construction_pass1=config.ef_construction_pass1,
        ef_construction_pass2=config.ef_construction_pass2,
        num_initializations=config.num_initializations,
        hubness_penalty_weight=config.hubness_penalty_weight,
        num_threads=config.num_threads,
        pass2_candidate_method=Pass2CandidateMethod.BEAM_SEARCH,
    )

    return index


def build_ensemble_index(
    data: np.ndarray,
    config: ExperimentConfig,
    variant: EnsembleVariant,
    distance_type: str = "l2"
) -> Any:
    """Build an ensemble index with the specified variant."""
    num_vectors, dim = data.shape

    index = flatnav_index.create_ensemble(
        distance_type=distance_type,
        dim=dim,
        dataset_size=num_vectors,
        data=data,
        variant=variant,
        num_graphs=config.num_graphs,
        M_per_graph=config.M_per_graph,
        M_final=config.M_final,
        ef_construction_per_graph=config.ef_construction_per_graph,
        ef_construction_final=config.ef_construction_final,
        num_initializations=config.num_initializations,
        hubness_penalty_weight=config.hubness_penalty_weight,
        num_threads=config.num_threads,
        seed_base=config.seed_base,
        use_neighbor_expansion=True,
        neighbor_expansion_hops=2,
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
    for target_recall in [0.90, 0.95]:
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
    method: str,
    variant: str,
    config: ExperimentConfig,
    distance_type: str = "l2"
) -> BenchmarkResult:
    """Run a single benchmark."""
    num_vectors, dim = data.shape

    print(f"  Building index: {method} ({variant})...")

    start_time = time.time()

    if method == "baseline":
        index = build_baseline_index(data, config, distance_type)
        M_final = config.M_baseline
        num_graphs = 1
        ef_total = config.ef_construction_baseline
    elif method == "two_pass":
        index = build_two_pass_index(data, config, distance_type)
        M_final = config.M_pass1 + config.M_pass2
        num_graphs = 1
        ef_total = config.ef_construction_pass1 + config.ef_construction_pass2
    elif method == "ensemble":
        variant_map = {
            "incremental": EnsembleVariant.MULTI_ORDER_INCREMENTAL,
            "reset": EnsembleVariant.MULTI_ORDER_RESET,
            "multi_seed": EnsembleVariant.MULTI_SEED,
        }
        ensemble_variant = variant_map.get(variant, EnsembleVariant.MULTI_ORDER_INCREMENTAL)
        index = build_ensemble_index(data, config, ensemble_variant, distance_type)
        M_final = config.M_final
        num_graphs = config.num_graphs
        ef_total = config.ef_construction_per_graph * config.num_graphs + config.ef_construction_final
    else:
        raise ValueError(f"Unknown method: {method}")

    construction_time = time.time() - start_time
    print(f"    Construction time: {construction_time:.2f}s")

    print(f"  Evaluating index...")
    eval_results = evaluate_index(index, queries, ground_truth, config)

    return BenchmarkResult(
        method=method,
        variant=variant,
        construction_time_seconds=construction_time,
        recall_at_1=eval_results.get("recall_at_1", 0.0),
        recall_at_10=eval_results.get("recall_at_10", 0.0),
        recall_at_100=eval_results.get("recall_at_100", 0.0),
        qps_at_90_recall=eval_results.get("qps_at_90_recall", 0.0),
        qps_at_95_recall=eval_results.get("qps_at_95_recall", 0.0),
        M_final=M_final,
        num_graphs=num_graphs,
        ef_construction_total=ef_total,
        num_vectors=num_vectors,
        dimension=dim
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble construction strategies"
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
        "--distance-type", type=str, default="l2",
        choices=["l2", "angular"],
        help="Distance metric to use"
    )
    parser.add_argument(
        "--output", type=str, default="ensemble_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--num-threads", type=int, default=8,
        help="Number of threads for construction"
    )
    parser.add_argument(
        "--M-baseline", type=int, default=16,
        help="M value for baseline single-pass"
    )
    parser.add_argument(
        "--M-final", type=int, default=16,
        help="Final M value for ensemble"
    )
    parser.add_argument(
        "--M-per-graph", type=int, default=8,
        help="M per graph in ensemble"
    )
    parser.add_argument(
        "--num-graphs", type=int, default=2,
        help="Number of graphs in ensemble"
    )
    parser.add_argument(
        "--ef-construction-baseline", type=int, default=100,
        help="ef_construction for baseline"
    )
    parser.add_argument(
        "--ef-construction-per-graph", type=int, default=25,
        help="ef_construction for each ensemble graph"
    )
    parser.add_argument(
        "--ef-construction-final", type=int, default=100,
        help="ef_construction for final pruning"
    )
    parser.add_argument(
        "--variants", type=str, nargs="+",
        default=["baseline", "two_pass", "incremental", "reset"],
        help="Variants to evaluate (baseline, two_pass, incremental, reset, multi_seed)"
    )
    parser.add_argument(
        "--hubness-penalty-weight", type=float, default=0.1,
        help="Weight for hubness penalty (default: 0.1)"
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
        M_pass1=args.M_per_graph,
        M_pass2=args.M_final - args.M_per_graph,
        ef_construction_pass1=args.ef_construction_per_graph,
        ef_construction_pass2=args.ef_construction_final,
        num_graphs=args.num_graphs,
        M_per_graph=args.M_per_graph,
        M_final=args.M_final,
        ef_construction_per_graph=args.ef_construction_per_graph,
        ef_construction_final=args.ef_construction_final,
        num_threads=args.num_threads,
        hubness_penalty_weight=args.hubness_penalty_weight,
    )

    # Map variant names to (method, variant) tuples
    variant_map = {
        "baseline": ("baseline", "single_pass"),
        "two_pass": ("two_pass", "re_prune_full"),
        "incremental": ("ensemble", "incremental"),
        "reset": ("ensemble", "reset"),
        "multi_seed": ("ensemble", "multi_seed"),
    }

    # Run benchmarks
    results = []
    for variant_name in args.variants:
        if variant_name not in variant_map:
            print(f"Unknown variant: {variant_name}, skipping...")
            continue

        method, variant = variant_map[variant_name]
        print(f"\nRunning benchmark: {variant_name}")

        result = run_benchmark(
            data, queries, ground_truth, method, variant, config, args.distance_type
        )

        results.append(asdict(result))
        print(f"  Recall@10: {result.recall_at_10:.4f}")
        print(f"  Construction time: {result.construction_time_seconds:.2f}s")

    # Print summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print(f"{'Method':<15} {'Variant':<15} {'Time (s)':<12} {'Recall@1':<12} {'Recall@10':<12} {'Recall@100':<12} {'Speedup':<12}")
    print("-" * 120)

    baseline_time = None
    for r in results:
        if r["method"] == "baseline":
            baseline_time = r["construction_time_seconds"]
            break

    for r in results:
        speedup = baseline_time / r["construction_time_seconds"] if baseline_time else 1.0
        print(f"{r['method']:<15} {r['variant']:<15} {r['construction_time_seconds']:<12.2f} "
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
