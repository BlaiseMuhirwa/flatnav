import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

BASE_PATH = "/root/data/speed-tests"
METRICS_PATH = "/root/metrics"

SYNTHETIC_DATASETS = [
    "normal-16-angular",
    "normal-16-euclidean",
    "normal-32-angular",
    "normal-32-euclidean",
    "normal-64-angular",
    "normal-64-euclidean",
    "normal-128-angular",
    "normal-128-euclidean",
    "normal-256-angular",
    "normal-256-euclidean",
    "normal-1024-angular",
    "normal-1024-euclidean",
    "normal-1536-angular",
    "normal-1536-euclidean",
]

ANN_DATASETS = [
    "glove-100-angular",
    "nytimes-256-angular",
    "gist-960-euclidean",
    "yandex-deep-10m-euclidean",
    "spacev-10m-euclidean",
]


def bin_data(query_flags: np.ndarray, num_bins: int = 10) -> list[np.ndarray]:
    query_len = len(query_flags)
    bin_size = query_len // num_bins
    binned_data = []
    for i in range(num_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < num_bins - 1 else query_len
        binned_data.append(query_flags[start:end])
    return binned_data


def plot_dataset(
    dataset_name: str,
    hub_percentages: np.ndarray,
    non_hub_percentages: np.ndarray,
    num_bins: int = 10,
) -> None:
    # Create the stacked bar chart
    bins = np.arange(1, num_bins + 1)

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjusting figure size

    ax.bar(bins, hub_percentages, label="Hub Nodes", color="blue")
    ax.bar(
        bins,
        non_hub_percentages,
        bottom=hub_percentages,
        label="Non-Hub Nodes",
        color="orange",
    )

    # Labels and titles
    ax.set_xlabel("Bins", fontsize=12)
    ax.set_ylabel("Percentage of Nodes Visited (%)", fontsize=12)
    ax.set_title(
        f"Percentage of Hub vs Non-Hub Nodes Visited. Dataset: {dataset_name}",
        fontsize=12,
    )

    # Improve x-axis labeling: Remove bin labels to avoid overlap
    ax.set_xticks(bins)
    ax.set_xticklabels(["" for _ in bins])  # Leave x-ticks empty

    # Add a single label for bins
    ax.set_xlabel("Bins", fontsize=12)

    ax.set_ylim([0, 100])  # Fix the y-axis to range from 0 to 100%
    ax.legend()

    # Save with higher resolution
    fig_name = os.path.join(METRICS_PATH, f"{dataset_name}_hubness_plot.png")
    plt.savefig(fig_name, dpi=300)  # Save figure with high resolution


def main():
    num_bins = 30
    for dataset in SYNTHETIC_DATASETS + ANN_DATASETS:
        visited_nodes_flags = np.load(
            os.path.join(BASE_PATH, f"{dataset}.99.0.npy"), allow_pickle=True
        )

        hub_percentages = np.zeros((num_bins,))
        non_hub_percentages = np.zeros((num_bins,))
        num_queries = len(visited_nodes_flags)

        # Process each query and aggregate the percentages
        for query_flags in visited_nodes_flags:
            print(f"Number of visited nodes: {len(query_flags)}")
            binned_data = bin_data(query_flags, num_bins)
            assert len(binned_data) == num_bins

            for i, bin_flags in enumerate(binned_data):
                num_hub_nodes = sum(bin_flags)  # Count of hub nodes in this bin
                bin_size = len(bin_flags)

                # Calculate percentage of hub and non-hub nodes in this bin
                hub_percentages[i] += (
                    (num_hub_nodes / bin_size) * 100 if bin_size > 0 else 0
                )
                non_hub_percentages[i] += (
                    ((bin_size - num_hub_nodes) / bin_size) * 100 if bin_size > 0 else 0
                )

        # Average the percentages over all queries
        hub_percentages /= num_queries
        non_hub_percentages /= num_queries

        plot_dataset(dataset, hub_percentages, non_hub_percentages, num_bins=num_bins)


if __name__ == "__main__":
    main()
