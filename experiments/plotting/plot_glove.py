import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.stats
import sys

plt.style.use('tableau-colorblind10')
hnsw_color = "#264478"
flatnav_color = "r"



def pareto_frontier(recall, latency):
    sorting_indices = recall.argsort()
    recall = recall[sorting_indices[::-1]]
    latency = latency[sorting_indices[::-1]]

    output_recall = []
    output_latency = []
    best_latency = latency[0]
    for l, r in zip(latency, recall):
        # descending recall order
        if l > best_latency:
            continue
        if l <= best_latency:
            best_latency = l
            output_recall.append(r)
            output_latency.append(l)
    return np.array(output_recall), np.array(output_latency)

def load_dataset(
    dataset_name,
    json_filename,
    latency_field,
):
    with open(json_filename) as f:
        d = json.load(f)
        latency = []
        recall = []
        for result in d[dataset_name]:
            latency.append(result[latency_field])
            recall.append(result["recall"])
        recall, latency = np.array(recall), np.array(latency)
        return pareto_frontier(recall, latency)


fig = plt.figure(figsize=(10,8))

plt.subplot(221)
dataset = "glove-25-angular"

hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p50")
plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW (P50)")
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p99")
plt.plot(hnsw_recall, hnsw_latency, 'x:', color = hnsw_color, label = "HNSW (P99)")

flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p50")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav (P50)")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p99")
plt.plot(flatnav_recall, flatnav_latency, 'x:', color = flatnav_color, label = "FlatNav (P99)")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 25$", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


plt.subplot(222)
dataset = "glove-50-angular"

hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p50")
plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW (P50)")
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p99")
plt.plot(hnsw_recall, hnsw_latency, 'x:', color = hnsw_color, label = "HNSW (P99)")

flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p50")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav (P50)")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p99")
plt.plot(flatnav_recall, flatnav_latency, 'x:', color = flatnav_color, label = "FlatNav (P99)")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 50$", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


plt.subplot(223)
dataset = "glove-100-angular"

hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p50")
plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW (P50)")
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p99")
plt.plot(hnsw_recall, hnsw_latency, 'x:', color = hnsw_color, label = "HNSW (P99)")

flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p50")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav (P50)")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p99")
plt.plot(flatnav_recall, flatnav_latency, 'x:', color = flatnav_color, label = "FlatNav (P99)")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 100$", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)



plt.subplot(224)
dataset = "glove-200-angular"

hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p50")
plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW (P50)")
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics-glove.json", "latency_p99")
plt.plot(hnsw_recall, hnsw_latency, 'x:', color = hnsw_color, label = "HNSW (P99)")

flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p50")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav (P50)")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics-glove.json", "latency_p99")
plt.plot(flatnav_recall, flatnav_latency, 'x:', color = flatnav_color, label = "FlatNav (P99)")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 200$", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


fig.supylabel('Latency', fontsize = 22)
fig.supxlabel('Recall (R100@100)', fontsize = 22)

plt.tight_layout()
plt.savefig("glove_results.png", dpi=400)
plt.show()

