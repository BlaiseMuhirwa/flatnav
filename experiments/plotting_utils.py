import seaborn as sns
import matplotlib.pyplot as plt


def plot_qps_against_recall(save_filepath: str, all_metrics: dict):
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
    plt.title("Recall-Queries per second Tradeoff - up and to the right is better")
    plt.legend()

    # Adjusting x-axis to use plain decimals instead of scientific notation
    plt.ticklabel_format(style="plain", axis="x")

    plt.savefig(save_filepath, dpi=300, bbox_inches="tight")
