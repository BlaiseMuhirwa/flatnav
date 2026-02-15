#!/usr/bin/env python3
"""
Plot Pareto frontiers from sweep results.

Reads a sweep results JSON (from evaluate_two_pass.py --sweep) and plots:
1. Construction time vs Recall@10
2. Distance computations vs Recall@10

Color-coded by strategy, with Pareto-optimal points annotated.

Usage:
    python plot_pareto.py --input sweep_results.json --output pareto.png
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np


STRATEGY_COLORS = {
    "baseline": "#1f77b4",   # blue
    "anchor": "#ff7f0e",     # orange
    "hubness": "#2ca02c",    # green
    "edge_quality": "#d62728",  # red
    "insertion_order": "#9467bd",  # purple
    "re_prune": "#8c564b",   # brown
}

STRATEGY_MARKERS = {
    "baseline": "o",
    "anchor": "s",
    "hubness": "^",
    "edge_quality": "D",
    "insertion_order": "v",
    "re_prune": "P",
}


def compute_pareto_front(
    points: List[Tuple[float, float]],
) -> List[int]:
    """
    Compute Pareto-optimal indices for minimizing y at each x level.

    For a Pareto frontier where x = recall (higher is better) and
    y = time or distance computations (lower is better), a point is
    Pareto-optimal if no other point has both higher x AND lower y.

    Args:
        points: List of (x, y) tuples where x = recall, y = cost

    Returns:
        List of indices into `points` that are Pareto-optimal
    """
    if not points:
        return []

    # Sort by x (recall) descending, then by y ascending
    indexed = sorted(enumerate(points), key=lambda t: (-t[1][0], t[1][1]))

    pareto_indices = []
    min_y = float("inf")

    for idx, (x, y) in indexed:
        if y <= min_y:
            pareto_indices.append(idx)
            min_y = y

    return sorted(pareto_indices)


def format_params(sweep_params: Dict[str, Any]) -> str:
    """Format sweep params into a short annotation string."""
    if not sweep_params:
        return ""
    parts = []
    for k, v in sweep_params.items():
        # Shorten key names
        short_key = k.replace("num_anchor_probes", "probes") \
                     .replace("bulk_ef_construction", "bef") \
                     .replace("ef_construction", "ef") \
                     .replace("num_initializations", "init")
        parts.append(f"{short_key}={v}")
    return ", ".join(parts)


def plot_pareto(results: List[Dict[str, Any]], output_path: str):
    """Plot two Pareto frontiers from sweep results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Group results by strategy
    by_strategy: Dict[str, List[Dict]] = {}
    for r in results:
        strategy = r["strategy"]
        by_strategy.setdefault(strategy, []).append(r)

    # --- Plot 1: Construction time vs Recall@10 ---
    ax1 = axes[0]
    for strategy, runs in by_strategy.items():
        color = STRATEGY_COLORS.get(strategy, "gray")
        marker = STRATEGY_MARKERS.get(strategy, "x")
        recalls = [r["recall_at_10"] for r in runs]
        times = [r["construction_time_seconds"] for r in runs]

        ax1.scatter(recalls, times, c=color, marker=marker, s=60,
                    label=strategy, alpha=0.7, edgecolors="black", linewidths=0.5)

        # Compute and plot Pareto front
        points = list(zip(recalls, times))
        pareto_idx = compute_pareto_front(points)
        if len(pareto_idx) >= 2:
            pareto_points = sorted([points[i] for i in pareto_idx])
            px, py = zip(*pareto_points)
            ax1.plot(px, py, c=color, linestyle="--", alpha=0.5, linewidth=1.5)

        # Annotate Pareto-optimal points
        for i in pareto_idx:
            params = runs[i].get("sweep_params")
            if params:
                label_text = format_params(params)
                ax1.annotate(
                    label_text, (recalls[i], times[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, alpha=0.8,
                )

    ax1.set_xlabel("Recall@10", fontsize=12)
    ax1.set_ylabel("Construction Time (seconds)", fontsize=12)
    ax1.set_title("Construction Time vs Recall@10", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Distance computations vs Recall@10 ---
    ax2 = axes[1]
    has_dist_data = False
    for strategy, runs in by_strategy.items():
        color = STRATEGY_COLORS.get(strategy, "gray")
        marker = STRATEGY_MARKERS.get(strategy, "x")

        # Filter to runs that have distance_computations
        valid_runs = [r for r in runs if r.get("distance_computations") is not None]
        if not valid_runs:
            continue
        has_dist_data = True

        recalls = [r["recall_at_10"] for r in valid_runs]
        dist_comps = [r["distance_computations"] for r in valid_runs]

        ax2.scatter(recalls, dist_comps, c=color, marker=marker, s=60,
                    label=strategy, alpha=0.7, edgecolors="black", linewidths=0.5)

        # Compute and plot Pareto front
        points = list(zip(recalls, dist_comps))
        pareto_idx = compute_pareto_front(points)
        if len(pareto_idx) >= 2:
            pareto_points = sorted([points[i] for i in pareto_idx])
            px, py = zip(*pareto_points)
            ax2.plot(px, py, c=color, linestyle="--", alpha=0.5, linewidth=1.5)

        # Annotate Pareto-optimal points
        for i in pareto_idx:
            params = valid_runs[i].get("sweep_params")
            if params:
                label_text = format_params(params)
                ax2.annotate(
                    label_text, (recalls[i], dist_comps[i]),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=6, alpha=0.8,
                )

    ax2.set_xlabel("Recall@10", fontsize=12)
    ax2.set_ylabel("Distance Computations", fontsize=12)
    ax2.set_title("Distance Computations vs Recall@10", fontsize=14)
    if has_dist_data:
        ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Pareto frontier plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot Pareto frontiers from sweep results"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to sweep results JSON file"
    )
    parser.add_argument(
        "--output", type=str, default="pareto.png",
        help="Output PNG file path (default: pareto.png)"
    )
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    results = data.get("results", data)
    if not isinstance(results, list):
        raise ValueError("Expected 'results' key with a list of benchmark results")

    plot_pareto(results, args.output)


if __name__ == "__main__":
    main()
