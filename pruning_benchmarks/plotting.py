import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import cycle
from scipy.ndimage import uniform_filter1d


# def pareto_frontier(x, y):
#     recall, latency = x, y  # To avoid refactoring the names.
#     sorting_indices = recall.argsort()
#     recall = recall[sorting_indices[::-1]]
#     latency = latency[sorting_indices[::-1]]

#     output_recall = []
#     output_latency = []
#     best_latency = latency[0]
#     for l, r in zip(latency, recall):
#         # descending recall order
#         if l > best_latency:
#             continue
#         if l <= best_latency:
#             best_latency = l
#             output_recall.append(r)
#             output_latency.append(l)
#     output_recall = np.array(output_recall)
#     output_latency = np.array(output_latency)
#     return output_recall[::-1], output_latency[::-1]


# def extract_data_series(data_dict, metric_name = "mean_distances"):
#     # Extracts the Pareto frontier of the data series for plotting.
#     mean_distances = []
#     mean_recalls = []
#     for efs, values in data_dict.items():
#         mean_distances.append(
#             values[metric_name]
#         )
#         mean_recalls.append(
#             values["recall"]
#         )
#     mean_distances = np.array(mean_distances)
#     mean_recalls = np.array(mean_recalls)
#     mean_recalls, mean_distances = pareto_frontier(mean_recalls, mean_distances)
#     return mean_recalls, mean_distances

# def interpolated_ratios(baseline_dict, candidate_dict, metric_name = "mean_distances"):
#     baseline_recall, baseline_metric = extract_data_series(baseline_dict, metric_name=metric_name)
#     candidate_recall, candidate_metric = extract_data_series(candidate_dict, metric_name=metric_name)

#     baseline_interp = np.interp(candidate_recall, baseline_recall, baseline_metric, left=np.nan, right=np.nan)
#     return candidate_recall, 100 * (candidate_metric / baseline_interp) - 100





# === STYLE ===
plt.style.use("seaborn-whitegrid")


# === CONFIGURATION ===
results_path = "pruning-results/glove100.json"
target_m_value = "M=16"
target_metric = "mean_latency"
dataset_title = "GLOVE100"
TOP_K_METHODS = 15
colors = cm.get_cmap("Set2", TOP_K_METHODS)

# === HELPER: Pareto Frontier ===
def pareto_frontier(X, Y, maximize_x=True, maximize_y=False):
    sorted_pairs = sorted(zip(X, Y), reverse=maximize_x)
    frontier_x = []
    frontier_y = []
    best_y = None

    for x, y in sorted_pairs:
        if best_y is None or (maximize_y and y > best_y) or (not maximize_y and y < best_y):
            frontier_x.append(x)
            frontier_y.append(y)
            best_y = y

    return np.array(frontier_x), np.array(frontier_y)

def extract_data_series(data_dict, metric_name="mean_distances"):
    recalls, metrics = [], []
    for efc_results in data_dict.values():
        for result in efc_results.values():
            recalls.append(result["recall"])
            val = result[metric_name]
            if "latency" in metric_name:
                val *= 1000
            metrics.append(val)
    recalls = np.array(recalls)
    metrics = np.array(metrics)

    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    metrics = metrics[sorted_indices]
    return pareto_frontier(recalls, metrics)

def interpolated_ratios(baseline_dict, candidate_dict, metric_name="mean_distances"):
    baseline_recall, baseline_metric = extract_data_series(baseline_dict, metric_name)
    candidate_recall, candidate_metric = extract_data_series(candidate_dict, metric_name)

    sorted_indices = np.argsort(candidate_recall)
    candidate_metric = candidate_metric[sorted_indices]
    baseline_recall = baseline_recall[sorted_indices]
    baseline_metric = baseline_metric[sorted_indices]

    if not np.all(np.diff(baseline_recall) > 0):
        print("baseline_recall is not sorted")
    if not np.all(np.diff(baseline_metric) > 0):
        print("baseline_metric is not sorted")

    print(f"Len of baseline_recall: {len(baseline_recall)}")
    print(f"Len of baseline_metric: {len(baseline_metric)}")
    print(f"Len of candidate_recall: {len(candidate_recall)}")
    print(f"Len of candidate_metric: {len(candidate_metric)}")


    baseline_interp = np.interp(
        candidate_recall,
        baseline_recall,
        baseline_metric,
        left=baseline_metric[0],
        right=baseline_metric[-1],
    )

    # baseline_interp = np.interp(candidate_recall, baseline_recall, baseline_metric, left=np.nan, right=np.nan)
    ratio_diff = 100 * (candidate_metric / baseline_interp) - 100
    return candidate_recall, ratio_diff

# === LOAD DATA ===
with open(results_path, "r") as f:
    data = json.load(f)

# === COMPUTE MEAN RECALL PER METHOD ===
recall_per_method = {}
for method_name, method_data in data.items():
    if target_m_value not in method_data:
        continue
    recalls = []
    for efc_results in method_data[target_m_value].values():
        for result in efc_results.values():
            recalls.append(result["recall"])
    if recalls:
        recall_per_method[method_name] = np.mean(recalls)

# === SELECT TOP METHODS (ensure arya_mount is first if present) ===
best_methods = sorted(recall_per_method, key=recall_per_method.get, reverse=True)[:TOP_K_METHODS]
if "arya_mount" in best_methods:
    best_methods.remove("arya_mount")
best_methods = ["arya_mount"] + best_methods

# === PLOTTING ===
fig, ax = plt.subplots(figsize=(10, 6))

for i, method_name in enumerate(best_methods):
    if target_m_value not in data[method_name]:
        continue
    method_data = data[method_name][target_m_value]

    print(f"Method: {method_name}")
    print(f"Method data: {method_data}")

    if method_name == "arya_mount":
        x, y = extract_data_series(method_data, metric_name=target_metric)
        ax.plot(
            x,
            [0] * len(x),
            linestyle="--",
            linewidth=2.5,
            color="black",
            label="arya_mount (baseline)",
            zorder=5,
        )
        continue

    try:
        recall_vals, percent_diff = interpolated_ratios(
            baseline_dict=data["arya_mount"][target_m_value],
            candidate_dict=method_data,
            metric_name=target_metric,
        )
    except Exception as e:
        print(f"Skipping {method_name} due to interpolation error: {e}")
        continue

    # Filter out any NaNs
    valid = ~np.isnan(percent_diff)
    recall_vals = recall_vals[valid]
    percent_diff = percent_diff[valid]
    if len(recall_vals) == 0:
        continue

    ax.plot(
        recall_vals,
        percent_diff,
        linestyle="-",
        linewidth=1.2,
        label=method_name,
        color=colors(i),
        alpha=0.9,
    )

    # Optional: annotate top-3 methods
    if i < 3:
        ax.annotate(
            method_name,
            xy=(recall_vals[-1], percent_diff[-1]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize="x-small",
            color=colors(i),
        )

# === BASELINE & LABELING ===
ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.set_xlabel("Mean Recall")
ylabel = f"{target_metric} (% difference from arya_mount)"
ax.set_ylabel(ylabel)
ax.set_title(f"{target_metric} vs. Mean Recall ({dataset_title}, {target_m_value})")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
ax.legend(loc="lower left", fontsize="small", ncol=2, frameon=False)

# === SAVE & SHOW ===
plt.tight_layout()
base_name = f"{dataset_title.lower()}_{target_metric}_vs_recall_{target_m_value}_pareto".replace("=", "")
plt.savefig(f"{base_name}.png", dpi=300, bbox_inches="tight")
plt.show()
