import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.stats
import sys

plt.style.use('tableau-colorblind10')

# datasets = ["gist-960-euclidean", "glove-100-angular", "normal-16-angular", "normal-16-euclidean", "normal-32-angular", "normal-32-euclidean", "normal-64-angular", "normal-64-euclidean", "normal-128-angular", "normal-128-euclidean", "normal-256-angular", "normal-256-euclidean", "normal-1024-angular", "normal-1024-euclidean", "normal-1536-angular", "normal-1536-euclidean", "nytimes-256-angular", "spacev-10m-euclidean", "yandex-deep-10m-euclidean"]

# plt.plot(q,gtruth,color = "k",alpha = 0.8,linewidth = 3,label = "Ground Truth")
# plt.plot(q,spectral,color = "#207CCC",alpha = 0.6, linestyle = ':',label = "Spectral")
# plt.plot(q,kme,color = '#6345A1',linestyle = '--', alpha = 0.6, label = "KME")
# plt.plot(q,pfda,color = "#003BFF",linestyle = (0, (5, 1)),label = "PFDA")
# plt.plot(q,bernstein,color = "#FF6600",linewidth = 1.75, linestyle = (0, (3, 1, 1, 1)),label = "Bernstein")
# plt.plot(q,privbayes,color = "orange",label = "PrivBayes")
# plt.plot(q,LMH,color = "green",label = "Synthetic")
# plt.plot(q,race,color = "red",label = "RACE")

###############################################################################
# SYNTHETIC DATA, ANGULAR DISTANCE
###############################################################################

plt.subplot(121)
datasets = ["normal-16-angular", "normal-32-angular","normal-64-angular","normal-128-angular", "normal-256-angular", "normal-1024-angular", "normal-1536-angular"]
labels = ["normal-16", "normal-32","normal-64","normal-128", "normal-256", "normal-1024", "normal-1536"]
# Cycle the prop cycle, because I hate the first color
plt.plot([])
# linestyles = {'solid':(0, ()),'loosely dotted':(0, (1.4, 10)),
# 'dotted':(0, (1, 5)),'densely dotted':(0, (1, 1)),'loosely dashed':(0, (5, 10)),
# 'dashed':(0, (5, 5)),'densely dashed':(0, (5, 1)),'loosely dashdotted':(0, (3, 10, 1, 10)),
# 'dashdotted':(0, (3, 5, 1, 5)),'densely dashdotted':(0, (3, 1, 1, 1)),
# 'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),'dashdotdotted':(0, (3, 5, 1, 5, 1, 5)),
# 'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}
# linestyles = ["-", "--", "-."]
linestyles = ["-"]
# datasets = ["gist-960-euclidean", "glove-100-angular", "normal-16-angular", "normal-16-euclidean", "normal-32-angular", "normal-32-euclidean", "normal-64-angular", "normal-64-euclidean", "normal-128-angular", "normal-128-euclidean", "normal-256-angular", "normal-256-euclidean", "normal-1024-angular", "normal-1024-euclidean", "normal-1536-angular", "normal-1536-euclidean", "nytimes-256-angular", "spacev-10m-euclidean", "yandex-deep-10m-euclidean"]
num_bins = 20
for idx, dataset in enumerate(datasets): #["normal-64-angular"]:
    print(f"Processing dataset {dataset}")
    sys.stdout.flush()
    node_access_filename = f"/scratch/brc7/node-access-distributions/{dataset}_node_access_counts.json"

    with open(node_access_filename) as f:
        d = json.load(f)
        accesses = np.zeros(len(d), dtype=float)
        for k, v in d.items():
            index = int(k)
            accesses[index] = v

    print(accesses)

    # Normalize insertions to a % of insertion.
    bins = np.geomspace(1, 10**4, num_bins)
    histogram, bins = np.histogram(accesses, bins=bins)
    label = labels[idx]
    linestyle = linestyles[idx % len(linestyles)]
    plt.plot(bins[:-1], histogram, linestyle=linestyle, label=label)

###############################################################################
# REAL DATA, ANGULAR DISTANCE
###############################################################################

labels = ["glove-100", "nytimes-256"]
datasets = ["glove-100-angular", "nytimes-256-angular"]
colors = ["k", "k"]
linestyles = ["-.", ":"]
num_bins = 20
for idx, dataset in enumerate(datasets): #["normal-64-angular"]:
    print(f"Processing dataset {dataset}")
    sys.stdout.flush()
    node_access_filename = f"node-access-distributions/{dataset}_node_access_counts.json"

    with open(node_access_filename) as f:
        d = json.load(f)
        accesses = np.zeros(len(d), dtype=float)
        for k, v in d.items():
            index = int(k)
            accesses[index] = v
    print(accesses)
    # Normalize insertions to a % of insertion.
    bins = np.geomspace(1, 10**4, num_bins)
    histogram, bins = np.histogram(accesses, bins=bins)
    label = labels[idx]
    color = colors[idx]
    linestyle = linestyles[idx % len(linestyles)]
    plt.plot(bins[:-1], histogram, linestyle=linestyle, color=color, label=label)


# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Log Access Count", fontsize = 22)
plt.ylabel("Log Frequency", fontsize = 22)
plt.title("Euclidean Distance", fontsize=24)
plt.legend(fontsize=12, ncol=2)


plt.subplot(122)

# Save the plot to a file
plt.savefig(f"{dataset}_nacd.png")



###############################################################################
# SYNTHETIC DATA, ANGULAR DISTANCE
###############################################################################

from scipy.stats import skew

# datasets = ["gist-960-euclidean", "glove-100-angular", "normal-16-angular", "normal-16-euclidean", "normal-32-angular", "normal-32-euclidean", "normal-64-angular", "normal-64-euclidean", "normal-128-angular", "normal-128-euclidean", "normal-256-angular", "normal-256-euclidean", "normal-1024-angular", "normal-1024-euclidean", "normal-1536-angular", "normal-1536-euclidean", "nytimes-256-angular", "spacev-10m-euclidean", "yandex-deep-10m-euclidean"]
datasets = [
    "normal-16-euclidean", "normal-32-euclidean","normal-64-euclidean",
    "normal-128-euclidean", "normal-256-euclidean", 
    "normal-1024-euclidean", "normal-1536-euclidean", "msmarco-384-euclidean", "yandex-deep-10m-euclidean", "spacev-10m-euclidean"
]
labels = ["normal-16", "normal-32","normal-64", "normal-128", "normal-256", "normal-1024", "normal-1536", "msmarco-384", "yandex-deep-10m", "spacev-10m"]
num_bins = 20
linestyles = ["-"]

# Style setup
plt.figure(figsize=(10, 7))
color_cycle = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
plt.rcParams.update({'font.size': 14})

# Cycle dummy to skip the first color
plt.plot([], [])

# Loop over datasets
for idx, dataset in enumerate(datasets):
    print(f"Processing dataset {dataset}")
    sys.stdout.flush()

    node_access_filename = f"/scratch/brc7/node-access-distributions/{dataset}_node_access_counts.json"
    with open(node_access_filename) as f:
        d = json.load(f)
        accesses = np.zeros(len(d), dtype=float)
        for k, v in d.items():
            index = int(k)
            accesses[index] = v

    # Compute histogram
    bins = np.geomspace(1, 10**6, num_bins)
    histogram, bins = np.histogram(accesses, bins=bins)

    # Compute skewness
    skew_val = skew(accesses)
    label = f"{labels[idx]} (skew={skew_val:.2f})"

    linestyle = linestyles[idx % len(linestyles)]
    plt.plot(bins[:-1], histogram, linestyle=linestyle, linewidth=2,
             label=label, color=color_cycle[idx])

# Log scaling and labels
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Log Access Count", fontsize=16)
plt.ylabel("Log Frequency", fontsize=16)
plt.title("Euclidean Distance", fontsize=20)

# Legend and grid
plt.legend(fontsize=11, ncol=2, frameon=False, loc='upper right')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Tight layout and save
plt.tight_layout()
plt.savefig("euclidean_nacd_skewed.png", dpi=300)

# ###############################################################################
# # REAL DATA, ANGULAR DISTANCE
# ###############################################################################

datasets = ["spacev-10m-euclidean", "yandex-deep-10m-euclidean"]
labels = ["spacev-10M", "Yandex-deep-10M"]
colors = ["k", "k"]
linestyles = ["-.", ":"]
num_bins = 20
for idx, dataset in enumerate(datasets): #["normal-64-angular"]:
    print(f"Processing dataset {dataset}")
    sys.stdout.flush()
    node_access_filename = f"node-access-distributions/{dataset}_node_access_counts.json"

    with open(node_access_filename) as f:
        d = json.load(f)
        accesses = np.zeros(len(d), dtype=float)
        for k, v in d.items():
            index = int(k)
            accesses[index] = v
    print(accesses)
    # Normalize insertions to a % of insertion.
    bins = np.geomspace(1, 10**6, num_bins)
    histogram, bins = np.histogram(accesses, bins=bins)
    label = labels[idx]
    color = colors[idx]
    linestyle = linestyles[idx % len(linestyles)]
    plt.plot(bins[:-1], histogram, linestyle=linestyle, color=color, label=label)



plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Log Access Count", fontsize = 22)
# plt.ylabel("Log Frequency", fontsize = 22)
plt.title("Euclidean Distance", fontsize=24)
plt.legend(fontsize=12, ncol=2)


plt.show()

