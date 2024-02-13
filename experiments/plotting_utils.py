import seaborn as sns
import matplotlib.pyplot as plt


def plot_qps_against_recall(save_filepath: str, all_metrics: dict, dataset_name: str):
    """
    Plot the QPS against recall for each key in all_metrics and save the plot to the specified filepath.
    :param save_filepath: The filepath to save the plot to
    :param all_metrics: A dictionary keyed on the index with values corresponding to
        a list of experiment runs. Each experiment run is a dictionary with keys "recall" and "qps"
    """

    markers = ["o", "X"]
    colors = sns.color_palette(n_colors=len(all_metrics))

    sns.set_theme(context="paper", style="darkgrid")
    plt.figure(figsize=(10, 6))

    for (color, marker), (key, metrics) in zip(
        zip(colors, markers), all_metrics.items()
    ):
        recall_values = [m["recall"] for m in metrics]
        qps_values = [m["qps"] for m in metrics]

        sns.lineplot(
            x=recall_values,
            y=qps_values,
            marker=marker,
            color=color,
            label=key,
            sort=True,
        )

    # Adding labels and title
    plt.xlabel("Recall")
    plt.ylabel("Queries per second (QPS)")
    plt.title(
        f"({dataset_name}) Recall-Queries per second Tradeoff - up and to the right is better"
    )
    plt.legend()

    # Adjusting x-axis to use plain decimals instead of scientific notation
    plt.ticklabel_format(style="plain", axis="x")

    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")


def plot_percentile_against_recall(
    save_filepath: str, all_metrics: dict, percentile_key: str, dataset_name: str
):
    """
    Plot the specified percentile latency against recall for each key in all_metrics
    and save the plot to the specified filepath.

    :param save_filepath: The filepath to save the plot to
    :param all_metrics: A dictionary keyed on the index with values corresponding to
        a list of experiment runs. Each experiment run is a dictionary with keys "recall"
        and the specified percentile_key, for example "latency_p95" or "latency_p99".
    :param percentile_key: The key for the percentile to plot against recall.
        This should match one of the keys in the dictionaries in all_metrics, for example "latency_p95" or "latency_p99".
    """

    # Make sure the percentile_key is valid for one of the dictionaries in all_metrics
    for metrics in all_metrics.values():
        if not all(percentile_key in m for m in metrics):
            raise ValueError(
                f"The specified percentile_key '{percentile_key}' is not present in all experiment runs."
            )

    markers = ["o", "X"]
    colors = sns.color_palette(n_colors=len(all_metrics))

    sns.set_theme(context="paper", style="darkgrid")
    plt.figure(figsize=(10, 6))

    for (color, marker), (key, metrics) in zip(
        zip(colors, markers), all_metrics.items()
    ):
        recall_values = [m["recall"] for m in metrics]
        percentile_values = [m[percentile_key] for m in metrics]

        sns.lineplot(
            x=recall_values,
            y=percentile_values,
            marker=marker,
            color=color,
            label=key,
            sort=True,
        )

    # Adding labels and title
    plt.xlabel("Recall")
    plt.ylabel(f"{percentile_key} (ms)")
    plot_title = f"({dataset_name}) {percentile_key} Latency-Recall Tradeoff - down and to the right is better"
    plt.title(
        plot_title
    )
    plt.legend()
    plt.ticklabel_format(style="plain", axis="x")
    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
