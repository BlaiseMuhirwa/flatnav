import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.stats
import sys

plt.style.use('tableau-colorblind10')
hnsw_color = "#264478"
flatnav_color = "r"

# From webplotdigitizer.

# uniform-4-10M-flatnav
dim_4_flatnav = [
(0.8067973856209151, 0.07647058823529414),
(0.8962091503267974, 0.0861764705882353),
(0.9155555555555556, 0.09058823529411769),
(0.956078431372549, 0.09941176470588235),
(0.9623529411764706, 0.11044117647058821),
(0.9811764705882353, 0.12014705882352943),
(0.9858823529411764, 0.13999999999999999),
(0.9890196078431372, 0.16955882352941182),
(0.9918954248366013, 0.27102941176470596),
]

# uniform-4-10M-hnsw
dim_4_hnsw = [
(0.7900653594771242, 0.008088235294117611),
(0.7911111111111111, 0.008529411764705855),
(0.9137254901960785, 0.008970588235294154),
(0.9981699346405228, 0.01382352941176454),
(0.9992156862745099, 0.018235294117646905),
(0.9992156862745099, 0.031911764705882306),
]

# uniform-8-10M-flatnav
dim_8_flatnav = [
(0.9251336898395722, 0.0348706896551724),
(0.981096256684492, 0.05017241379310344),
(0.9942914438502674, 0.06353448275862068),
(0.998783422459893, 0.08745689655172413),
(0.9995320855614973, 0.11116379310344827),
(0.9998128342245989, 0.14607758620689654),
]

# uniform-8-10M-hnsw
dim_8_hnsw = [
(0.9638770053475936, 0.01784482758620684),
(0.9797860962566844, 0.01870689655172407),
(0.9946657754010695, 0.025818965517241395),
(0.9963502673796791, 0.02711206896551721),
(0.9976604278074866, 0.027327586206896553),
(0.998783422459893, 0.033577586206896555),
(0.9991577540106952, 0.035301724137931034),
(0.9994385026737967, 0.03573275862068964),
(0.9998128342245989, 0.048663793103448275),
(0.9999064171122994, 0.05211206896551725),
]

# uniform-16-10M-flatnav
dim_16_flatnav = [
(0.3694117647058824, 0.02978593272171256),
(0.5121568627450981, 0.03051987767584099),
(0.6972549019607843, 0.03076452599388381),
(0.699607843137255, 0.04788990825688075),
(0.7976470588235294, 0.048868501529052),
(0.8533333333333334, 0.04935779816513762),
(0.8690196078431374, 0.06452599388379206),
(0.9129411764705884, 0.06574923547400616),
(0.9380392156862745, 0.09388379204892967),
(0.9631372549019608, 0.096085626911315),
(0.9662745098039217, 0.12226299694189603),
(0.9811764705882352, 0.12568807339449542),
(0.9850980392156863, 0.16507645259938836),
(0.9937254901960784, 0.169480122324159),
]

# uniform-16-10M-hnsw
dim_16_hnsw = [
(0.5482352941176472, 0.023425076452599395),
(0.6345098039215686, 0.025137614678899092),
(0.7113725490196079, 0.026360856269113164),
(0.7545098039215687, 0.027094801223241594),
(0.7937254901960784, 0.03834862385321103),
(0.8564705882352943, 0.04006116207951071),
(0.8870588235294119, 0.04128440366972476),
(0.9160784313725491, 0.05253822629969418),
(0.9372549019607843, 0.05498470948012235),
(0.9631372549019608, 0.0782262996941896),
(0.9756862745098039, 0.08091743119266055),
(0.979607843137255, 0.10293577981651378),
(0.987450980392157, 0.10660550458715598),
(0.9905882352941177, 0.1398776758409786),
(0.9945098039215686, 0.14525993883792046),
]

# uniform-32-10M-flatnav
dim_32_flatnav = [
(0.11955922865013772, 0.05357142857142792),
(0.18457300275482091, 0.053571428571427965),
(0.2672176308539945, 0.053571428571427916),
(0.4292011019283747, 0.0750000000000006),
(0.5195592286501378, 0.10714285714285728),
(0.5349862258953169, 0.10714285714285726),
(0.6528925619834711, 0.14999999999999905),
(0.6639118457300275, 0.16071428571428603),
(0.7377410468319561, 0.2035714285714282),
(0.7476584022038568, 0.2035714285714283),
(0.81267217630854, 0.24642857142857025),
(0.8236914600550964, 0.2678571428571431),
(0.8997245179063362, 0.47142857142857153),
(0.9338842975206614, 0.4714285714285706),
(0.9460055096418734, 0.6857142857142852),
(0.9669421487603307, 0.6857142857142845),
(0.9801652892561983, 1.114285714285714),
(0.9889807162534436, 1.1142857142857145),
(0.9966942148760332, 2.153571428571429),
(1, 6.332142857142858),
]

# uniform-32-10M-hnsw
dim_32_hnsw = [
(0.12947658402203854, 0.053571428571427944),
(0.23856749311294767, 0.05357142857142803),
(0.3862258953168044, 0.08571428571428559),
(0.48760330578512395, 0.11785714285714272),
(0.5261707988980717, 0.12857142857142836),
(0.7333333333333332, 0.2035714285714286),
(0.7454545454545456, 0.2035714285714282),
(0.8137741046831957, 0.2464285714285714),
(0.824793388429752, 0.25714285714285734),
(0.9261707988980716, 0.48214285714285665),
(0.9316804407713499, 0.5035714285714287),
(0.9647382920110192, 0.6964285714285703),
(0.9867768595041324, 1.1250000000000004),
(0.9933884297520663, 2.196428571428572),
(0.9977961432506888, 2.1964285714285716),
(0.9988980716253444, 6.396428571428571),
(0.9988980716253444, 6.450000000000001),
]

def load_dataset(recall_latency):
    recall = []
    latency = []
    for (r, l) in recall_latency:
        recall.append(r)
        latency.append(l)
    return np.array(recall), np.array(latency)

fig = plt.figure(figsize=(10,8))

plt.subplot(221)
hnsw_recall, hnsw_latency = load_dataset(dim_4_hnsw)
flatnav_recall, flatnav_latency = load_dataset(dim_4_flatnav)

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 4$", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


plt.subplot(222)
hnsw_recall, hnsw_latency = load_dataset(dim_8_hnsw)
flatnav_recall, flatnav_latency = load_dataset(dim_8_flatnav)

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 8$", fontsize=24)
plt.grid(which='both')
plt.legend(fontsize=12, ncol=1)


plt.subplot(223)
hnsw_recall, hnsw_latency = load_dataset(dim_16_hnsw)
flatnav_recall, flatnav_latency = load_dataset(dim_16_flatnav)

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 16$", fontsize=24)
plt.legend(fontsize=12, ncol=1)
plt.grid(which='both')


plt.subplot(224)
hnsw_recall, hnsw_latency = load_dataset(dim_32_hnsw)
flatnav_recall, flatnav_latency = load_dataset(dim_32_flatnav)

plt.plot(hnsw_recall, hnsw_latency, 'x-', color = hnsw_color, label = "HNSW")
plt.plot(flatnav_recall, flatnav_latency, 'x-', color = flatnav_color, label = "FlatNav")

# We change the fontsize of minor ticks label 
plt.gca().tick_params(axis='both', which='major', labelsize=12)
plt.gca().tick_params(axis='both', which='minor', labelsize=8)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel("", fontsize = 22)
# plt.ylabel("P50 Latency", fontsize = 22)
plt.title("$d = 32$", fontsize=24)
plt.legend(fontsize=12, ncol=1)
plt.grid(which='both')

fig.supylabel('P50 Latency (ms)', fontsize = 22)
fig.supxlabel('Recall (R1@1)', fontsize = 22)
plt.suptitle("Synthetic Uniform", fontsize = 24)

plt.tight_layout()
plt.savefig("lin_and_zhao.png", dpi=400)
plt.show()
