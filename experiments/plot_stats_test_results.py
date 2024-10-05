# Data for plotting
datasets = {
    "normal-16-angular": {
        "mann-whitney-p-value": 0.49829919627199387,
        "two-sample-t-test-p-value": 0.45706087893147584,
        "effect_size": 0.0048258243223664125
    },
    "normal-16-euclidean": {
        "mann-whitney-p-value": 0.6103672711932897,
        "two-sample-t-test-p-value": 0.35063374637858924,
        "effect_size": 0.01716667999411063
    },
    "normal-32-angular": {
        "mann-whitney-p-value": 0.41337579450665773,
        "two-sample-t-test-p-value": 0.2861712945721231,
        "effect_size": 0.025266797673224668
    },
    "normal-32-euclidean": {
        "mann-whitney-p-value": 0.8176089216561105,
        "two-sample-t-test-p-value": 0.8135538913668252,
        "effect_size": -0.03987871452583527
    },
    "normal-64-angular": {
        "mann-whitney-p-value": 0.17017818154209252,
        "two-sample-t-test-p-value": 0.22326178892645315,
        "effect_size": 0.034066722547644396
    },
    "normal-64-euclidean": {
        "mann-whitney-p-value": 0.9175780095130485,
        "two-sample-t-test-p-value": 0.9435362712126693,
        "effect_size": -0.07095793973733296
    },
    "normal-128-angular": {
        "mann-whitney-p-value": 0.4864893646024787,
        "two-sample-t-test-p-value": 0.6655415565157279,
        "effect_size": -0.01913683936450564
    },
    "normal-128-euclidean": {
        "mann-whitney-p-value": 0.45340654877518755,
        "two-sample-t-test-p-value": 0.46978377134443583,
        "effect_size": 0.0033926027700751384
    },
    "normal-256-angular": {
        "mann-whitney-p-value": 0.11903131028542119,
        "two-sample-t-test-p-value": 0.19999402198087085,
        "effect_size": 0.03766628602476098
    },
    "normal-256-euclidean": {
        "mann-whitney-p-value": 0.11386043573393156,
        "two-sample-t-test-p-value": 0.07120992225475872,
        "effect_size": 0.06565786921794088
    },
    "normal-1024-angular": {
        "mann-whitney-p-value": 0.41229289178432726,
        "two-sample-t-test-p-value": 0.35111375700947667,
        "effect_size": 0.017108671159425135
    },
    "normal-1024-euclidean": {
        "mann-whitney-p-value": 0.07529944092745168,
        "two-sample-t-test-p-value": 0.022100949286267775,
        "effect_size": 0.09008967715207827
    },
    "normal-1536-angular": {
        "mann-whitney-p-value": 0.06187600431969492,
        "two-sample-t-test-p-value": 0.04745949212507695,
        "effect_size": 0.07475774838361963
    },
    "normal-1536-euclidean": {
        "mann-whitney-p-value": 0.1563097852544691,
        "two-sample-t-test-p-value": 0.1690825244366408,
        "effect_size": 0.04286570847818611
    },
    "glove-100-angular": {
        "mann-whitney-p-value": 0.6721651977457639,
        "two-sample-t-test-p-value": 0.6545151004477228,
        "effect_size": -0.01778998359204667
    },
    "nytimes-256-angular": {
        "mann-whitney-p-value": 0.10541233073193851,
        "two-sample-t-test-p-value": 0.5391860349240402,
        "effect_size": -0.004402607682992809
    },
    "gist-960-euclidean": {
        "mann-whitney-p-value": 0.11158327264486029,
        "two-sample-t-test-p-value": 0.031249045641264843,
        "effect_size": 0.08339836588164547
    },
    "yandex-deep-10m-euclidean": {
        "mann-whitney-p-value": 0.5001154828900115,
        "two-sample-t-test-p-value": 0.5,
        "effect_size": 0.0
    },
    "spacev-10m-euclidean": {
        "mann-whitney-p-value": 0.02269608361974771,
        "two-sample-t-test-p-value": 0.02272301128813566,
        "effect_size": 0.08962214074009246
    }
}

import numpy as np 
import matplotlib.pyplot as plt
import os 

BASE_DIR = "/root/metrics"

os.makedirs(BASE_DIR, exist_ok=True)

# Prepare the data for plotting
dataset_names = list(datasets.keys())
mann_whitney_p_values = [datasets[name]["mann-whitney-p-value"] for name in dataset_names]
t_test_p_values = [datasets[name]["two-sample-t-test-p-value"] for name in dataset_names]
effect_sizes = [datasets[name]["effect_size"] for name in dataset_names]

# Plotting
x = np.arange(len(dataset_names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, mann_whitney_p_values, width, label='Mann-Whitney U-Test p-value', color='blue')
rects2 = ax.bar(x + width/2, t_test_p_values, width, label='Two-Sample T-Test p-value', color='orange')

# Set the y-axis to a logarithmic scale to better visualize smaller p-values
ax.set_yscale('log')

# Add a horizontal line at the threshold for significance (p-value = 0.05)
ax.axhline(y=0.05, color='red', linestyle='--', label='p-value = 0.05')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('P-Value (log scale)')
ax.set_title('P-Values by Dataset and Test Type (Log Scale) with Significance Threshold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha="right")
ax.legend()

# Annotate bars with effect size
for i, rect in enumerate(rects1):
    height = rect.get_height()
    ax.annotate(f'{effect_sizes[i]:.2f}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Display the plot
plt.tight_layout()

# Save figure
path = os.path.join(BASE_DIR, "p_values_by_dataset_and_test_type.png")
plt.savefig(path)

