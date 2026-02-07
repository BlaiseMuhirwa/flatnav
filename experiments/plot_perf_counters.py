#!/usr/bin/env python3
"""
Plot hardware performance counter comparisons between construction strategies.

Reads the results JSON from evaluate_two_pass.py (which contains perf_counters
data per strategy) and generates grouped bar charts comparing key cache and
CPU metrics.

Usage:
    python plot_perf_counters.py \
        --input results.json \
        --output perf_comparison.png

    # Or use a dedicated perf counters JSON:
    python plot_perf_counters.py \
        --input perf_counters.json \
        --perf-only \
        --output perf_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Metrics to plot: (metric_key, display_label, section_key_pattern)
# section_key_pattern: the perf counter section name to look for
METRIC_CONFIGS = [
    {
        "title": "L1D Cache Miss Rate (%)",
        "metric": "l1d_miss_rate",
        "ylabel": "Miss Rate (%)",
    },
    {
        "title": "L1D Misses per Call",
        "metric": "l1d_misses_per_call",
        "ylabel": "Misses / Call",
    },
    {
        "title": "dTLB Misses per Call",
        "metric": "dtlb_misses_per_call",
        "ylabel": "Misses / Call",
    },
    {
        "title": "Cycles per Call",
        "metric": "cycles_per_call",
        "ylabel": "Cycles / Call",
    },
    {
        "title": "Instructions per Cycle (IPC)",
        "metric": "ipc",
        "ylabel": "IPC",
    },
    {
        "title": "LLC Misses per Call",
        "metric": "llc_misses_per_call",
        "ylabel": "Misses / Call",
    },
]


def load_perf_data(input_path: str, perf_only: bool = False):
    """Load perf counter data from results JSON.

    Returns:
        dict: {strategy_name: {section_name: {metric: value}}}
    """
    with open(input_path) as f:
        data = json.load(f)

    if perf_only:
        # Direct perf counters JSON: {strategy -> {section -> {metric -> value}}}
        return data

    # Full results JSON: extract perf_counters from each result entry
    perf_data = {}
    for result in data.get("results", []):
        strategy = result.get("strategy", "unknown")
        counters = result.get("perf_counters")
        if counters:
            perf_data[strategy] = counters

    return perf_data


def plot_perf_comparison(perf_data, output_path: str):
    """Generate grouped bar charts comparing strategies across metrics."""
    strategies = list(perf_data.keys())
    if not strategies:
        print("No perf counter data found in input file.")
        return

    # Collect all section names across strategies
    all_sections = set()
    for strategy_data in perf_data.values():
        all_sections.update(strategy_data.keys())
    sections = sorted(all_sections)

    if not sections:
        print("No perf counter sections found.")
        return

    # Determine layout
    num_metrics = len(METRIC_CONFIGS)
    cols = 2
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(strategies), 3)))

    for idx, metric_cfg in enumerate(METRIC_CONFIGS):
        ax = axes[idx]
        metric_key = metric_cfg["metric"]

        # For each section, group strategies
        x = np.arange(len(sections))
        width = 0.8 / max(len(strategies), 1)

        for s_idx, strategy in enumerate(strategies):
            values = []
            for section in sections:
                section_data = perf_data.get(strategy, {}).get(section, {})
                values.append(section_data.get(metric_key, 0.0))

            offset = (s_idx - len(strategies) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=strategy,
                   color=colors[s_idx % len(colors)])

        ax.set_title(metric_cfg["title"], fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_cfg["ylabel"], fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(sections, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Performance Counter Comparison: Construction Strategies",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot perf counter comparisons between construction strategies"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to results JSON (from evaluate_two_pass.py) or dedicated perf JSON"
    )
    parser.add_argument(
        "--perf-only", action="store_true",
        help="Input is a dedicated perf counters JSON (from --perf-output)"
    )
    parser.add_argument(
        "--output", type=str, default="perf_comparison.png",
        help="Output plot file path (default: perf_comparison.png)"
    )

    args = parser.parse_args()

    perf_data = load_perf_data(args.input, perf_only=args.perf_only)
    plot_perf_comparison(perf_data, args.output)


if __name__ == "__main__":
    main()
