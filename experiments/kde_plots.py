
import json
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from scipy.stats import skew 
import os

DISTRIBUTIONS_SAVE_PATH = "/root/node-access-distributions"
PLOTS_SAVE_PATH = "/root/node-access-distributions"

os.makedirs(PLOTS_SAVE_PATH, exist_ok=True)

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


def plot_kde_distributions(distributions: dict, typename: str, save_path: str, bw_adjust_value=0.7):
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()  # Get the current Axes instance

    # Compute skewness before the log transformation and store them
    skewness_values = {}

    for dataset_name, node_access_counts in distributions.items():
        node_access_count_values = list(node_access_counts.values())
        # Map this list to convert all values to integers
        node_access_count_values = list(map(int, node_access_count_values))
        skewness_values[dataset_name] = skew(node_access_count_values)

    for dataset_name, node_access_counts in distributions.items():
        node_access_count_values = list(node_access_counts.values())
        # Map this list to convert all values to integers
        node_access_count_values = list(map(int, node_access_count_values))
        # Apply log-transform to the counts, adding 1 to avoid log(0)
        log_counts = np.log1p(node_access_count_values)
        raw_skewness = skewness_values[dataset_name]
        # Replace "euclidean" with "l2" and "angular" with "cosine"
        dataset_name = dataset_name.replace("euclidean", "l2").replace(
            "angular", "cosine"
        )

        print(f"Log counts: {log_counts}")

        # Plot the KDE for log-transformed data with less smoothness
        sns.kdeplot(
            log_counts,
            label=f"{dataset_name} ($\\tilde{{\\mu}}_3$ = {raw_skewness:.4f})",
            bw_adjust=bw_adjust_value,
        )

    # Set up the legend on the right of the plot
    plt.legend(title="Dataset", loc="center left", bbox_to_anchor=(1, 0.5))

    # Improve plot aesthetics
    sns.despine(trim=True)  # Trim the spines for a cleaner look
    plt.grid(True)  # Add gridlines
    plt.xlabel("Log of Node access counts")
    plt.ylabel("PDF")
    plt.title("KDE of Node Access Counts")

    # Adjust the plot area to fit the legend and increase the resolution
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()

    filename = os.path.join(save_path, f"distributions_{typename}_{bw_adjust_value}.png")
    plt.savefig(filename)

    print(f"Saved plot to {filename}")


def main():
    distributions = {}
    datasets = SYNTHETIC_DATASETS + ANN_DATASETS

    for dataset in datasets:
        print(f"Loading {dataset}...")
        path = os.path.join(DISTRIBUTIONS_SAVE_PATH, f"{dataset}_node_access_counts.json")
        with open(path, "r") as f:
            distributions[dataset] = json.load(f)

        # Length of dataset 
        length = len(distributions[dataset])
        print(f"Type of {dataset}: {type(distributions[dataset])}")
        print(f"Length of {dataset}: {length}")
        
        # Print teh first 100 values 
        # print(f"First 100 values of {dataset}: {list(distributions[dataset].values())[:100]}")

        # exit(0)
        

    # Create four separate sets from the distributions dictionary based on first whether 
    # the dataset is synthetic or ANN, and then whether it is angular or euclidean
    synthetic_angular = {k: v for k, v in distributions.items() if "angular" in k and "synthetic" in k}
    synthetic_euclidean = {k: v for k, v in distributions.items() if "euclidean" in k and "synthetic" in k}
    ann_angular = {k: v for k, v in distributions.items() if "angular" in k and "ann" in k}
    ann_euclidean = {k: v for k, v in distributions.items() if "euclidean" in k and "ann" in k}

    # Plot the KDE distributions for each set
    plot_kde_distributions(synthetic_angular, "synthetic_angular", PLOTS_SAVE_PATH)
    plot_kde_distributions(synthetic_euclidean, "synthetic_l2", PLOTS_SAVE_PATH)
    plot_kde_distributions(ann_angular, "ann_angular", PLOTS_SAVE_PATH)
    plot_kde_distributions(ann_euclidean, "ann_l2", PLOTS_SAVE_PATH)


if __name__ == "__main__":
    main()