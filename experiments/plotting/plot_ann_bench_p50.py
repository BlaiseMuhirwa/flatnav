import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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




# fig = plt.figure(figsize=(10,8))
fig = plt.figure(figsize=(10,12))

gs = gridspec.GridSpec(6, 4)



ax = plt.subplot(gs[0:2, 0:2])
dataset = "deep-image-96"
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics_ann_bench.json", "latency_p50")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics_ann_bench.json", "latency_p50")

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("DEEP", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


ax = plt.subplot(gs[0:2, 2:4])
dataset = "gist"
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics_ann_bench.json", "latency_p50")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics_ann_bench.json", "latency_p50")

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("GIST", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


ax = plt.subplot(gs[2:4, 0:2])
dataset = "mnist-784"
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics_ann_bench.json", "latency_p50")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics_ann_bench.json", "latency_p50")

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("MNIST", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


ax = plt.subplot(gs[2:4, 2:4])
dataset = "nytimes-256-angular"
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics_ann_bench.json", "latency_p50")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics_ann_bench.json", "latency_p50")

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("NY-Times (Angular)", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)



ax = plt.subplot(gs[4:6, 1:3])
dataset = "sift"
hnsw_recall, hnsw_latency = load_dataset(f"{dataset}_hnsw", "metrics_ann_bench.json", "latency_p50")
flatnav_recall, flatnav_latency = load_dataset(f"{dataset}_flatnav", "metrics_ann_bench.json", "latency_p50")

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("SIFT", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


fig.supylabel('P50 Latency (ms)', fontsize = 28)
fig.supxlabel('Recall (R100@100)', fontsize = 28)

plt.tight_layout()
plt.savefig("ann_bench_p50.png", dpi=400)
# plt.show()






