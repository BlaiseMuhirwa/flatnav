import json
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
results_path = "pruning-results/sift.json"  # <- change as needed
target_m_value = "M=16"                     # <- options: "M=16", "M=32"
target_metric = "p50_latency"              # <- or "p99_latency", "p90_distances", etc.
dataset_title = "SIFT"

def generate_distinct_colors(n):
    import matplotlib.colors as mcolors
    hsv_colors = [(i / n, 0.75, 0.75) for i in range(n)]
    return [mcolors.hsv_to_rgb(hsv) for hsv in hsv_colors]

# === LOAD DATA ===
with open(results_path, "r") as f:
    data = json.load(f)

# === COMPUTE MAX RECALL PER METHOD ===
recall_per_method = {}
for method_name, method_data in data.items():
    if target_m_value not in method_data:
        continue
    m_results = method_data[target_m_value]

    recalls = []
    for efc_key in sorted(m_results.keys(), key=lambda k: int(k.split("=")[-1])):
        efc_results = m_results[efc_key]
        for ef_key in sorted(efc_results.keys(), key=lambda k: int(k.split("=")[-1])):
            result = efc_results[ef_key]
            recalls.append(result["recall"])
    
    if recalls:
        recall_per_method[method_name] = max(recalls)

# === FILTER OUT BOTTOM 2 METHODS ===
worst_methods = sorted(recall_per_method, key=recall_per_method.get)[:2]
print("Filtered out (worst recall):", worst_methods)

# === PLOT SETUP ===
fig, ax = plt.subplots(figsize=(14, 9))
filtered_methods = [m for m in data if m not in worst_methods]
color_list = generate_distinct_colors(len(filtered_methods))

# === PLOT EACH METHOD ===
for i, method_name in enumerate(filtered_methods):
    method_data = data[method_name]
    if target_m_value not in method_data:
        continue
    m_results = method_data[target_m_value]

    recalls = []
    raw_latency_ms = []

    for efc_key in sorted(m_results.keys(), key=lambda k: int(k.split("=")[-1])):
        efc_results = m_results[efc_key]
        for ef_key in sorted(efc_results.keys(), key=lambda k: int(k.split("=")[-1])):
            result = efc_results[ef_key]
            recall = result["recall"]
            latency_ms = result[target_metric] * 1000  # Convert to milliseconds

            recalls.append(recall)
            raw_latency_ms.append(latency_ms)

    if not recalls:
        continue

    # === Normalize latency per method ===
    min_latency = min(raw_latency_ms)
    norm_latency = [v / min_latency for v in raw_latency_ms]

    # Sort by recall
    zipped = sorted(zip(recalls, norm_latency))
    recalls_sorted, latency_sorted = zip(*zipped)

    ax.plot(
        recalls_sorted,
        latency_sorted,
        marker="x",
        linestyle="-",
        label=method_name,
        color=color_list[i],
    )

# === LABELS AND LEGEND ===
ax.set_xlabel("Mean Recall")
ax.set_ylabel(f"Normalized {target_metric} (ms, min=1.0)")
ax.set_title(f"Normalized {target_metric} vs. Mean Recall ({dataset_title} Dataset, {target_m_value})")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0., fontsize="small")
plt.tight_layout(rect=[0, 0, 0.85, 1])

# === SAVE OR SHOW ===
output_file = f"normalized_{target_metric}_vs_recall_{target_m_value}.png".replace("=", "")
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.show()
