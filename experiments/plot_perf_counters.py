#!/usr/bin/env python3
"""
Plot hardware performance counter comparisons between construction strategies.

Reads the results JSON from evaluate_two_pass.py (which contains perf_counters
data per strategy) and generates grouped bar charts comparing the primary
entry point selection method of each strategy.

For each strategy, the "primary" entry point section is the one with the most
calls (e.g., baseline uses initializeSearch for all points; anchor uses
findEntryViaAnchors for the bulk pass).

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

import matplotlib.pyplot as plt
import numpy as np


METRIC_CONFIGS = [
    {
        "title": "L1D Loads per Call",
        "metric": "l1d_loads_per_call",
        "ylabel": "Loads / Call",
    },
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
        "title": "L2 Misses per Call",
        "metric": "l2_misses_per_call",
        "ylabel": "Misses / Call",
    },
    {
        "title": "DRAM Fills per Call (L3 Miss Proxy)",
        "metric": "dram_fills_per_call",
        "ylabel": "Fills / Call",
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
]


def load_perf_data(input_path: str, perf_only: bool = False):
    """Load perf counter data from results JSON.

    Returns:
        dict: {strategy_name: {section_name: {metric: value}}}
    """
    with open(input_path) as f:
        data = json.load(f)

    if perf_only:
        return data

    perf_data = {}
    for result in data.get("results", []):
        strategy = result.get("strategy", "unknown")
        counters = result.get("perf_counters")
        if counters:
            perf_data[strategy] = counters

    # Backfill derived per-call metrics that may be absent in older data
    for strategy_data in perf_data.values():
        for section_data in strategy_data.values():
            n = section_data.get("call_count", 0)
            if n > 0:
                if "l1d_loads_per_call" not in section_data and "l1d_loads" in section_data:
                    section_data["l1d_loads_per_call"] = section_data["l1d_loads"] / n

    return perf_data


def pick_primary_section(sections: dict) -> tuple:
    """Pick the entry point section with the most calls.

    Returns:
        (section_name, section_data) for the highest call_count section.
    """
    best_name, best_data, best_count = None, None, -1
    for name, data in sections.items():
        count = data.get("call_count", 0)
        if count > best_count:
            best_name, best_data, best_count = name, data, count
    return best_name, best_data


def format_number(val):
    """Format a number for bar annotation."""
    if val == 0:
        return "0"
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    if abs(val) >= 1:
        return f"{val:.1f}"
    return f"{val:.3f}"


def plot_perf_comparison(perf_data, output_path: str):
    """Generate bar charts comparing the primary entry point method per strategy."""
    if not perf_data:
        print("No perf counter data found in input file.")
        return

    # For each strategy, pick its primary entry point section
    bars = []  # list of (label, section_data)
    for strategy in sorted(perf_data.keys()):
        section_name, section_data = pick_primary_section(perf_data[strategy])
        if section_data is None:
            continue
        call_count = int(section_data.get("call_count", 0))
        label = f"{strategy}\n({section_name}, {call_count:,} calls)"
        bars.append((label, section_data))

    if not bars:
        print("No perf counter sections found.")
        return

    labels = [b[0] for b in bars]
    num_bars = len(labels)

    # Filter out metrics that are all zero
    active_metrics = []
    for mcfg in METRIC_CONFIGS:
        key = mcfg["metric"]
        if any(b[1].get(key, 0) != 0 for b in bars):
            active_metrics.append(mcfg)

    if not active_metrics:
        print("All metrics are zero — nothing to plot.")
        return

    num_metrics = len(active_metrics)
    cols = 2
    rows = (num_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 + 2.5 * num_bars, 4 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

    for idx, metric_cfg in enumerate(active_metrics):
        ax = axes[idx]
        metric_key = metric_cfg["metric"]
        values = [b[1].get(metric_key, 0.0) for b in bars]

        x = np.arange(num_bars)
        bar_colors = [colors[i % len(colors)] for i in range(num_bars)]
        rects = ax.bar(x, values, width=0.5, color=bar_colors, edgecolor="white",
                       linewidth=0.8)

        # Annotate bars with values
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.annotate(
                format_number(val),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

        # Show ratio annotation between bars (if exactly 2)
        if num_bars == 2 and values[0] != 0 and values[1] != 0:
            # For IPC, higher is better; for everything else, lower is better
            higher_is_better = "ipc" in metric_key
            if higher_is_better:
                if values[0] >= values[1]:
                    ratio = values[0] / values[1]
                    ratio_text = f"{ratio:.1f}x higher"
                else:
                    ratio = values[1] / values[0]
                    ratio_text = f"{ratio:.1f}x higher"
            else:
                if values[0] <= values[1]:
                    ratio = values[1] / values[0]
                    ratio_text = f"{ratio:.1f}x fewer"
                else:
                    ratio = values[0] / values[1]
                    ratio_text = f"{ratio:.1f}x more"
            mid_y = max(values) * 0.5
            ax.annotate(
                ratio_text, xy=(0.5, mid_y),
                ha="center", va="center", fontsize=9, fontstyle="italic",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                          edgecolor="#cccccc", alpha=0.9),
            )

        ax.set_title(metric_cfg["title"], fontsize=11, fontweight="bold")
        ax.set_ylabel(metric_cfg["ylabel"], fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "Entry Point Selection: Per-Call Cache & CPU Metrics",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

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
