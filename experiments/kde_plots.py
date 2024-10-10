
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

def silverman_bandwidth(data: np.ndarray) -> float:
    """
    Compute the bandwidth using Silverman's Rule of Thumb.

    :param data: Input array of data points
    :return: Computed bandwidth (h)
    """
    n = len(data)
    if n == 0:
        return 1.0  # Return a default value if no data is provided to avoid division by zero

    std_dev = np.std(data)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    
    # Silverman's rule of thumb
    bandwidth = 0.9 * min(std_dev, iqr / 1.35) * n ** (-1 / 5)
    
    return bandwidth

def scott_bandwidth(data: np.ndarray) -> float:
    """
    Compute the bandwidth using Scott's Rule of Thumb.

    :param data: Input array of data points
    :return: Computed bandwidth (h)
    """
    n = len(data)
    if n == 0:
        return 1.0  # Return a default value if no data is provided to avoid division by zero

    std_dev = np.std(data)
    
    # Scott's rule of thumb
    bandwidth = std_dev * n ** (-1 / 5)

    return 10 * bandwidth



def plot_kde_distributions(distributions: dict, typename: str, save_path: str, bandwidth: int = 1.5):
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

        # Compute Silverman bandwidth for the current dataset
        # bandwidth = scott_bandwidth(log_counts)
        bandwidth = bandwidth

        raw_skewness = skewness_values[dataset_name]
        # Replace "euclidean" with "l2" and "angular" with "cosine"
        dataset_name = dataset_name.replace("euclidean", "l2").replace("angular", "cosine")

        print(f"Bandwidth for {dataset_name}: {bandwidth}")

        # Plot the KDE for log-transformed data with a dataset-specific bandwidth
        sns.kdeplot(
            log_counts,
            label=f"{dataset_name} ($\\tilde{{\\mu}}_3$ = {raw_skewness:.4f})",
            bw_adjust=bandwidth  # Use computed bandwidth
        )

    # Set up the legend on the right of the plot
    plt.legend(title="Dataset", loc="center left", bbox_to_anchor=(1, 0.5))

    # Improve plot aesthetics
    sns.despine(trim=True)  # Trim the spines for a cleaner look
    plt.grid(True)  # Add gridlines
    plt.xlabel("Log of Node access counts")
    plt.ylabel("PDF")
    plt.title(f"KDE of Node Access Counts - {typename}")

    # Adjust the plot area to fit the legend and increase the resolution
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()

    filename = os.path.join(save_path, f"distributions_{typename}.png")
    plt.savefig(filename)


def main():
    angular_datasets, euclidean_datasets = {}, {}

    for dataset in ANN_DATASETS:
        print(f"Loading {dataset}...")
        path = os.path.join(DISTRIBUTIONS_SAVE_PATH, f"{dataset}_node_access_counts.json")
        with open(path, "r") as f:
            if "angular" in dataset:
                # angular_datasets[dataset] = json.load(f)
                pass 
            else:
                euclidean_datasets[dataset] = json.load(f)

    for dataset in SYNTHETIC_DATASETS:
        print(f"Loading {dataset}...")
        path = os.path.join(DISTRIBUTIONS_SAVE_PATH, f"{dataset}_node_access_counts.json")
        with open(path, "r") as f:
            if "angular" in dataset:
                # angular_datasets[dataset] = json.load(f)
                pass 
            else:
                euclidean_datasets[dataset] = json.load(f)

    # Plot the KDE distributions for each set with individual bandwidth values
    # plot_kde_distributions(angular_datasets, "angular", PLOTS_SAVE_PATH, bandwidth=0.9)
    plot_kde_distributions(euclidean_datasets, "l2", PLOTS_SAVE_PATH, bandwidth=0.3)

if __name__ == "__main__":
    main()