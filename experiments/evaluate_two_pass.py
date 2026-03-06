#!/usr/bin/env python3
"""
Evaluate two-pass construction strategies on large-scale benchmarks.

This script compares:
- Baseline: Single-pass construction with M=8
- Two-pass strategies: M=4 + M=4 with different optimization strategies
- Anchor construction: Build high-quality graph on a small subset, use it
  as a navigation layer for bulk insertion

Hypothesis: Two-pass with M=4+M=4 achieves similar recall to single-pass M=8
but with faster construction time due to cheaper graph maintenance.
Anchor construction should be fastest at large scale due to avoiding random
initializeSearch across the full dataset.

Usage:
    python evaluate_two_pass.py \
        --dataset /path/to/train.npy \
        --queries /path/to/queries.npy \
        --gtruth /path/to/ground_truth.npy \
        --strategies baseline anchor hubness \
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
from flatnav import (
    TwoPassStrategy, Pass2CandidateMethod,
    TwoPassConfig, AnchorConfig, SQConfig, DataType,
    build,
    reset_perf_counters, get_perf_counters,
)
from data_loader import get_data_loader

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
    distance_computations: Optional[int] = None
    perf_counters: Optional[Dict[str, Any]] = None
    # Sweep parameters (filled in when --sweep is active)
    sweep_params: Optional[Dict[str, Any]] = None


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

    # Anchor construction parameters
    anchor_fraction: float = 0.01       # Fraction of data as anchors (1%)
    anchor_M: int = 16                  # M for anchor index (same for all nodes)
    anchor_ef_construction: int = 500   # ef for anchor graph (high quality)
    bulk_ef_construction: int = 80      # ef for bulk insertion
    num_anchor_probes: int = 10         # Number of anchor probes for entry point finding
    anchor_seed: int = 42               # Random seed for anchor selection

    # Scalar quantization parameters
    use_sq_quantization: bool = False
    sq_quant_max_train_samples: int = 0

    # Index data type
    index_data_type: DataType = DataType.float32

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

    return build(
        distance_type=distance_type,
        dim=dim,
        dataset_size=num_vectors,
        data=data,
        max_edges_per_node=config.M_baseline,
        ef_construction=config.ef_construction_baseline,
        num_initializations=config.num_initializations,
        num_threads=config.num_threads,
        index_data_type=config.index_data_type,
    )


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
        "re_prune": TwoPassStrategy.RE_PRUNE_FULL,
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

    return build(
        distance_type=distance_type,
        dim=dim,
        dataset_size=num_vectors,
        data=data,
        config=TwoPassConfig(
            strategy=strategy_enum,
            M_pass1=config.M_pass1,
            M_pass2=config.M_pass2,
            ef_construction_pass1=config.ef_construction_pass1,
            ef_construction_pass2=config.ef_construction_pass2,
            hubness_penalty_weight=config.hubness_penalty_weight,
            edge_quality_threshold=config.edge_quality_threshold,
            pass2_candidate_method=candidate_method,
            neighbor_expansion_hops=config.neighbor_expansion_hops,
        ),
        num_initializations=config.num_initializations,
        num_threads=config.num_threads,
        index_data_type=config.index_data_type,
    )


def build_anchor_index(
    data: np.ndarray,
    config: ExperimentConfig,
    distance_type: str = "l2"
) -> Any:
    """Build an index using anchor-based two-pass construction.

    Pass 1: Build a high-quality graph on anchor_fraction of the data.
    Pass 2: Insert remaining data using the anchor graph for entry point finding.
    """
    num_vectors, dim = data.shape

    quantization = None
    if config.use_sq_quantization:
        quantization = SQConfig(
            training_data=data,
            max_train_samples=config.sq_quant_max_train_samples,
        )

    return build(
        distance_type=distance_type,
        dim=dim,
        dataset_size=num_vectors,
        data=data,
        config=AnchorConfig(
            anchor_fraction=config.anchor_fraction,
            M=config.anchor_M,
            anchor_ef_construction=config.anchor_ef_construction,
            bulk_ef_construction=config.bulk_ef_construction,
            num_anchor_probes=config.num_anchor_probes,
            seed=config.anchor_seed,
            quantization=quantization,
        ),
        num_initializations=config.num_initializations,
        num_threads=config.num_threads,
        index_data_type=config.index_data_type,
    )



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
    distance_type: str = "l2",
    save_index_prefix: str = None,
) -> BenchmarkResult:
    """Run a single benchmark for a given strategy."""
    num_vectors, dim = data.shape

    print(f"  Building index with strategy: {strategy}...")

    reset_perf_counters()

    start_time = time.time()
    if strategy == "baseline":
        index = build_baseline_index(data, config, distance_type)
        M_total = config.M_baseline
        ef_total = config.ef_construction_baseline
    elif strategy == "anchor":
        index = build_anchor_index(data, config, distance_type)
        M_total = config.anchor_M
        ef_total = config.anchor_ef_construction
    else:
        index = build_two_pass_index(data, config, strategy, distance_type)
        M_total = config.M_pass1 + config.M_pass2
        ef_total = config.ef_construction_pass1 + config.ef_construction_pass2
    construction_time = time.time() - start_time

    # Read distance computations accumulated during construction
    distance_comps = index.get_distance_computations()

    perf_data = get_perf_counters()

    print(f"    Construction time: {construction_time:.2f}s")
    print(f"    Distance computations: {distance_comps:,}")

    if save_index_prefix is not None:
        save_path = f"{save_index_prefix}-{strategy}.bin"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"    Saving index to {save_path}...")
        index.save(save_path)
        print(f"    Index saved. Check size with: du -h {save_path}")

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
        dimension=dim,
        distance_computations=distance_comps,
        perf_counters=perf_data if perf_data else None,
    )


def run_sweep(
    data: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    strategies: List[str],
    config: ExperimentConfig,
    distance_type: str,
    sweep_anchor_probes: List[int],
    sweep_bulk_ef: List[int],
    sweep_baseline_ef: List[int],
    sweep_baseline_init: List[int],
) -> List[BenchmarkResult]:
    """Run parameter sweep across strategies, returning all results."""
    from itertools import product

    results = []

    for strategy in strategies:
        if strategy == "baseline":
            combos = list(product(sweep_baseline_ef, sweep_baseline_init))
            for ef_val, init_val in combos:
                sweep_config = ExperimentConfig(
                    M_baseline=config.M_baseline,
                    ef_construction_baseline=ef_val,
                    num_initializations=init_val,
                    num_threads=config.num_threads,
                    ef_search_values=config.ef_search_values,
                    K=config.K,
                )
                params = {"ef_construction": ef_val, "num_initializations": init_val}
                print(f"\n[sweep] baseline: ef={ef_val}, init={init_val}")
                result = run_benchmark(data, queries, ground_truth, strategy, sweep_config, distance_type)
                result.sweep_params = params
                results.append(result)

        elif strategy == "anchor":
            combos = list(product(sweep_anchor_probes, sweep_bulk_ef))
            for probes, bulk_ef in combos:
                sweep_config = ExperimentConfig(
                    anchor_fraction=config.anchor_fraction,
                    anchor_M=config.anchor_M,
                    anchor_ef_construction=config.anchor_ef_construction,
                    bulk_ef_construction=bulk_ef,
                    num_anchor_probes=probes,
                    num_initializations=config.num_initializations,
                    num_threads=config.num_threads,
                    anchor_seed=config.anchor_seed,
                    ef_search_values=config.ef_search_values,
                    K=config.K,
                )
                params = {"num_anchor_probes": probes, "bulk_ef_construction": bulk_ef}
                print(f"\n[sweep] anchor: probes={probes}, bulk_ef={bulk_ef}")
                result = run_benchmark(data, queries, ground_truth, strategy, sweep_config, distance_type)
                result.sweep_params = params
                results.append(result)

        else:
            # For two-pass strategies, just run with the default config
            print(f"\n[sweep] {strategy}: default config")
            result = run_benchmark(data, queries, ground_truth, strategy, config, distance_type)
            results.append(result)

    return results


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
        "--range", type=int, required=False,
        default=None,
        nargs="+",
        help="The first element is the start index and the second element is the end index. Must be two integers.",
    )
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=["baseline", "hubness", "edge_quality", "insertion_order", "re_prune", "anchor"],
        help="Strategies to evaluate (baseline, hubness, edge_quality, insertion_order, re_prune, anchor)"
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

    # Anchor construction arguments
    parser.add_argument(
        "--anchor-fraction", type=float, default=0.01,
        help="Fraction of data to use as anchors (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--anchor-M", type=int, default=32,
        help="M (edges per node) for anchor construction (default: 32)"
    )
    parser.add_argument(
        "--anchor-ef-construction", type=int, default=500,
        help="ef_construction for anchor graph (default: 500)"
    )
    parser.add_argument(
        "--bulk-ef-construction", type=int, default=80,
        help="ef_construction for bulk insertion in anchor mode (default: 80)"
    )
    parser.add_argument(
        "--num-anchor-probes", type=int, default=10,
        help="Number of anchor probes for initial seeding (default: 10)"
    )
    parser.add_argument(
        "--anchor-seed", type=int, default=42,
        help="Random seed for anchor selection (default: 42)"
    )
    parser.add_argument(
        "--index-data-type", type=str, default="float32",
        choices=["float32", "int8", "uint8"],
        help="Data type for index storage (default: float32)"
    )

    # Scalar quantization arguments
    parser.add_argument(
        "--use-sq-quantization", action="store_true",
        help="Apply scalar quantization on top of anchor construction"
    )
    parser.add_argument(
        "--sq-quant-max-train-samples", type=int, default=0,
        help="Max training samples for SQ (0 = use all, default: 0)"
    )
    parser.add_argument(
        "--perf-output", type=str, default=None,
        help="Output file for dedicated perf counter results JSON"
    )

    # Index saving
    parser.add_argument(
        "--save-index", type=str, default=None,
        help="Path prefix to save constructed indices (e.g. /tmp/sift). "
             "Each strategy saves as <prefix>-<strategy>.bin"
    )

    # Sweep mode arguments
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run parameter sweep mode: iterate over multiple parameter configurations per strategy"
    )
    parser.add_argument(
        "--sweep-anchor-probes", type=int, nargs="+", default=[5, 10, 20, 50],
        help="Anchor probe counts to sweep (default: 5 10 20 50)"
    )
    parser.add_argument(
        "--sweep-bulk-ef", type=int, nargs="+", default=[50, 80, 100],
        help="Bulk ef_construction values to sweep for anchor strategy (default: 50 80 100)"
    )
    parser.add_argument(
        "--sweep-baseline-ef", type=int, nargs="+", default=[50, 80, 100],
        help="ef_construction values to sweep for baseline strategy (default: 50 80 100)"
    )
    parser.add_argument(
        "--sweep-baseline-init", type=int, nargs="+", default=[50, 100],
        help="num_initializations values to sweep for baseline strategy (default: 50 100)"
    )

    args = parser.parse_args()

    if args.range is not None:
        data_loader = get_data_loader(
            train_dataset_path=args.dataset,
            queries_path=args.queries,
            ground_truth_path=args.gtruth,
            range=args.range,
        )
        data, queries, ground_truth = data_loader.load_data()
    else:
        data = np.load(args.dataset)
        queries = np.load(args.queries)
        ground_truth = np.load(args.gtruth)

    # Print the shape of each numpy array above 
    print(f"  Shape of data: {data.shape}")
    print(f"  Shape of queries: {queries.shape}")
    print(f"  Shape of ground_truth: {ground_truth.shape}")

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
        anchor_fraction=args.anchor_fraction,
        anchor_M=args.anchor_M,
        anchor_ef_construction=args.anchor_ef_construction,
        bulk_ef_construction=args.bulk_ef_construction,
        num_anchor_probes=args.num_anchor_probes,
        anchor_seed=args.anchor_seed,
        use_sq_quantization=args.use_sq_quantization,
        sq_quant_max_train_samples=args.sq_quant_max_train_samples,
        index_data_type=getattr(DataType, args.index_data_type),
    )

    # Run benchmarks
    if args.sweep:
        # Sweep mode: iterate over parameter combinations
        sweep_results = run_sweep(
            data, queries, ground_truth, args.strategies, config, args.distance_type,
            sweep_anchor_probes=args.sweep_anchor_probes,
            sweep_bulk_ef=args.sweep_bulk_ef,
            sweep_baseline_ef=args.sweep_baseline_ef,
            sweep_baseline_init=args.sweep_baseline_init,
        )
        results = [asdict(r) for r in sweep_results]
    else:
        results = []
        for strategy in args.strategies:
            print(f"\nRunning benchmark with strategy: {strategy}")

            result = run_benchmark(
                data, queries, ground_truth, strategy, config, args.distance_type,
                save_index_prefix=args.save_index,
            )

            results.append(asdict(result))
            print(f"  Recall@10: {result.recall_at_10:.4f}")
            print(f"  Construction time: {result.construction_time_seconds:.2f}s")

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    header = (f"{'Strategy':<20} {'Time (s)':<12} {'Speedup':<10} {'Recall@1':<12} {'Recall@10':<12} "
              f"{'Recall@100':<12} {'Dist Comps':<15}")
    print(header)
    print("-" * 123)

    baseline_time = None
    for r in results:
        if r["strategy"] == "baseline":
            baseline_time = r["construction_time_seconds"]
            break

    for r in results:
        speedup = baseline_time / r["construction_time_seconds"] if baseline_time else 1.0
        dist_comps = r.get("distance_computations")
        dist_str = f"{dist_comps:,}" if dist_comps is not None else "N/A"
        speedup_str = f"{speedup:.2f}x" if baseline_time and r["strategy"] != "baseline" else "-"
        print(f"{r['strategy']:<20} {r['construction_time_seconds']:<12.2f} {speedup_str:<10} "
              f"{r['recall_at_1']:<12.4f} {r['recall_at_10']:<12.4f} "
              f"{r['recall_at_100']:<12.4f} {dist_str:<15}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "config": asdict(config),
            "results": results
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # Save dedicated perf counter output if requested
    if args.perf_output:
        perf_output_path = Path(args.perf_output)
        perf_output_path.parent.mkdir(parents=True, exist_ok=True)
        perf_results = {
            r["strategy"]: r.get("perf_counters", {})
            for r in results
        }
        with open(perf_output_path, "w") as f:
            json.dump(perf_results, f, indent=2)
        print(f"Perf counter results saved to {perf_output_path}")


if __name__ == "__main__":
    main()
