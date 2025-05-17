import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.stats
import sys


datasets = ["gist-960-euclidean", "glove-100-angular", "normal-16-angular", "normal-16-euclidean", "normal-32-angular", "normal-32-euclidean", "normal-64-angular", "normal-64-euclidean", "normal-128-angular", "normal-128-euclidean", "normal-256-angular", "normal-256-euclidean", "normal-1024-angular", "normal-1024-euclidean", "normal-1536-angular", "normal-1536-euclidean", "nytimes-256-angular", "spacev-10m-euclidean", "yandex-deep-10m-euclidean"]


class Plotter():
    def __init__(self, ax):
        self.ax = ax

    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs, color = 'k', alpha = 0.4)

    def text(self, *args, **kwargs):
        return


for dataset in datasets: #["normal-64-angular"]:
    print(f"Processing dataset {dataset}")
    sys.stdout.flush()
    insertion_order_filename = f"insertion-orders/{dataset}_insertion_order.json"
    node_access_filename = f"node-access-distributions/{dataset}_node_access_counts.json"

    with open(insertion_order_filename) as f:
        d = json.load(f)
        insertions = np.zeros(len(d), dtype=float)
        for k, v in d.items():
            index = int(k)
            insertions[index] = v

    with open(node_access_filename) as f:
        d = json.load(f)
        accesses = np.zeros(len(d), dtype=float)
        for k, v in d.items():
            index = int(k)
            accesses[index] = v

    # print(accesses)
    # print(insertions)

    # Normalize insertions to a % of insertion.
    insertions = insertions / np.max(insertions)
    insertions = np.log(insertions + 1)
    accesses = np.log(accesses + 1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(insertions, accesses)
    print(f"Eqtn: y = {slope} x + {intercept}")
    print("R^2: ", r_value**2)
    print("p: ", p_value)

    residuals = (slope * insertions + intercept) - accesses
    scipy.stats.probplot(residuals[::100], dist="norm", plot=Plotter(plt.gca()), fit=False)
    # plt.show()
    sys.stdout.flush()

plt.show()

    # plt.plot(insertions, accesses, 'o', alpha = 0.01)
    # plt.xlabel("Insertion order (% of max))", fontsize = 22)
    # plt.ylabel("Hubness (Node access count)", fontsize = 22)
    # plt.title(dataset, fontsize=24)
    # plt.show()

'''
Processing dataset gist-960-euclidean
Eqtn: y = 0.9564236913243112 x + 6.0427241543378445
R^2:  0.00041069421905353645
p:  2.4824137873331604e-91
Processing dataset glove-100-angular
Eqtn: y = 4.121406539110042 x + 55.2249325573131
R^2:  0.0003820445268620595
p:  2.347074134383437e-100
Processing dataset normal-16-angular
Eqtn: y = -9.772006788145251 x + 69.71401939407264
R^2:  0.017419325730923238
p:  0.0
Processing dataset normal-16-euclidean
Eqtn: y = -0.007392283127719639 x + 64.85545614156386
R^2:  2.0599789299361724e-09
p:  0.9637988953766118
Processing dataset normal-32-angular
Eqtn: y = -15.911165087778974 x + 73.6282385438895
R^2:  0.06226066881576832
p:  0.0
Processing dataset normal-32-euclidean
Eqtn: y = 10.147702719745297 x + 60.30616464012735
R^2:  0.0008622903006564432
p:  1.2860299043309934e-189
Processing dataset normal-64-angular
Eqtn: y = -35.115964730407164 x + 88.40916636520359
R^2:  0.16018955284783085
p:  0.0
Processing dataset normal-64-euclidean
Eqtn: y = 22.37670716248091 x + 54.835246418759546
R^2:  0.0008633741795898023
p:  7.471620705722116e-190
Processing dataset normal-128-angular
Eqtn: y = -60.40925602644421 x + 103.83638801322212
R^2:  0.22393846950673038
p:  0.0
Processing dataset normal-128-euclidean
Eqtn: y = 37.93773165440848 x + 47.065966172795754
R^2:  0.0010270782658431572
p:  1.796139896185295e-225
Processing dataset normal-256-angular
Eqtn: y = -73.91480928455489 x + 110.72686064227746
R^2:  0.21758008788243244
p:  0.0
Processing dataset normal-256-euclidean
Eqtn: y = 43.65207242657169 x + 43.86501978671416
R^2:  0.0005212064768400456
p:  2.1616216517823274e-115
Processing dataset normal-1024-angular
Eqtn: y = -83.3256751462295 x + 114.96650157311475
R^2:  0.18622290578430137
p:  0.0
Processing dataset normal-1024-euclidean
Eqtn: y = 44.08940424538785 x + 43.66225787730607
R^2:  0.0005577525308786817
p:  2.398697218851703e-123
Processing dataset normal-1536-angular
Eqtn: y = -84.27715572874463 x + 115.47436986437232
R^2:  0.1790325795254548
p:  0.0
Processing dataset normal-1536-euclidean
Eqtn: y = 49.455607177240985 x + 40.884116411379495
R^2:  0.0005938289959113911
p:  3.3729503441604717e-131
Processing dataset nytimes-256-angular
Eqtn: y = -33.68850563473665 x + 252.616335575989
R^2:  0.003802365295368703
p:  3.022618831059302e-242
Processing dataset spacev-10m-euclidean
Eqtn: y = -0.8762500173956029 x + 20.486170208697803
R^2:  7.131387032859619e-05
p:  4.1034906216231674e-157
Processing dataset yandex-deep-10m-euclidean
Eqtn: y = 0.05006192078828413 x + 6.553786639605858
R^2:  2.7578953596411486e-06
p:  1.5080841196745643e-07

gist-960-euclidean, < 0.1%
glove-100-angular, < 0.1%
normal-16-angular, 1.7%
normal-16-euclidean, < 0.1%
normal-32-angular, 6.2%
normal-32-euclidean, < 0.1%
normal-64-angular, 16.0%
normal-64-euclidean, < 0.1%
normal-128-angular, 22.4%
normal-128-euclidean, < 0.1%
normal-256-angular, 21.8%
normal-256-euclidean, < 0.1%
normal-1024-angular, 18.6%
normal-1024-euclidean, < 0.1%
normal-1536-angular, 17.9%
normal-1536-euclidean, < 0.1%
nytimes-256-angular, 0.4%
spacev-10m-euclidean, < 0.1%
yandex-deep-10m-euclidean, < 0.1%
'''
