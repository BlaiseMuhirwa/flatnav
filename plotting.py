import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# --- Paste the cleaned JSON data here ---
data = {
  "arya_mount": {
    "100": {
      "recall": 0.592510999999999,
      "mean_distances": 653.4467
    },
    "200": {
      "recall": 0.7156659999999916,
      "mean_distances": 1074.0157
    },
    "300": {
      "recall": 0.7783429999999874,
      "mean_distances": 1478.3185
    },
    "400": {
      "recall": 0.8168189999999826,
      "mean_distances": 1868.8512
    },
    "500": {
      "recall": 0.8432789999999768,
      "mean_distances": 2248.1651
    },
    "600": {
      "recall": 0.862394999999972,
      "mean_distances": 2617.8994
    },
    "700": {
      "recall": 0.8772299999999679,
      "mean_distances": 2979.9253
    },
    "800": {
      "recall": 0.8888999999999673,
      "mean_distances": 3335.0085
    },
    "900": {
      "recall": 0.8984079999999656,
      "mean_distances": 3684.3755
    },
    "1000": {
      "recall": 0.9063159999999593,
      "mean_distances": 4027.867
    }
  },
  "arya_mount_reproduction": {
    "100": {
      "recall": 0.592153999999998,
      "mean_distances": 653.051
    },
    "200": {
      "recall": 0.715709999999994,
      "mean_distances": 1074.7913
    },
    "300": {
      "recall": 0.7782839999999892,
      "mean_distances": 1479.1868
    },
    "400": {
      "recall": 0.8170299999999857,
      "mean_distances": 1869.8281
    },
    "500": {
      "recall": 0.8434279999999821,
      "mean_distances": 2249.1252
    },
    "600": {
      "recall": 0.8628999999999766,
      "mean_distances": 2619.5712
    },
    "700": {
      "recall": 0.8777929999999695,
      "mean_distances": 2981.8077
    },
    "800": {
      "recall": 0.8897549999999672,
      "mean_distances": 3337.464
    },
    "900": {
      "recall": 0.8991219999999627,
      "mean_distances": 3687.0099
    },
    "1000": {
      "recall": 0.9069609999999607,
      "mean_distances": 4030.8742
    }
  },
  "cheap_outdegree_conditional_m_over_4": {
    "100": {
      "recall": 0.5931069999999989,
      "mean_distances": 654.2719
    },
    "200": {
      "recall": 0.7173279999999925,
      "mean_distances": 1076.1081
    },
    "300": {
      "recall": 0.7799469999999898,
      "mean_distances": 1480.4642
    },
    "400": {
      "recall": 0.8182819999999849,
      "mean_distances": 1872.1375
    },
    "500": {
      "recall": 0.8446359999999747,
      "mean_distances": 2252.2849
    },
    "600": {
      "recall": 0.8640149999999698,
      "mean_distances": 2623.6087
    },
    "700": {
      "recall": 0.8787749999999711,
      "mean_distances": 2986.705
    },
    "800": {
      "recall": 0.8906089999999669,
      "mean_distances": 3343.4063
    },
    "900": {
      "recall": 0.9001809999999617,
      "mean_distances": 3694.0116
    },
    "1000": {
      "recall": 0.908049999999959,
      "mean_distances": 4038.6541
    }
  },
  "cheap_outdegree_conditional_2": {
    "100": {
      "recall": 0.591955,
      "mean_distances": 653.813
    },
    "200": {
      "recall": 0.7168999999999948,
      "mean_distances": 1077.0775
    },
    "300": {
      "recall": 0.7794049999999884,
      "mean_distances": 1481.403
    },
    "400": {
      "recall": 0.8177989999999823,
      "mean_distances": 1872.7058
    },
    "500": {
      "recall": 0.844402999999978,
      "mean_distances": 2253.0208
    },
    "600": {
      "recall": 0.8635519999999735,
      "mean_distances": 2624.3702
    },
    "700": {
      "recall": 0.8784469999999704,
      "mean_distances": 2987.5303
    },
    "800": {
      "recall": 0.8901459999999662,
      "mean_distances": 3343.9036
    },
    "900": {
      "recall": 0.8998119999999646,
      "mean_distances": 3694.5943
    },
    "1000": {
      "recall": 0.9078299999999581,
      "mean_distances": 4039.6564
    }
  },
  "cheap_outdegree_conditional_4": {
    "100": {
      "recall": 0.5932619999999995,
      "mean_distances": 661.9003
    },
    "200": {
      "recall": 0.7196759999999938,
      "mean_distances": 1091.3875
    },
    "300": {
      "recall": 0.7843169999999894,
      "mean_distances": 1503.8949
    },
    "400": {
      "recall": 0.8233529999999813,
      "mean_distances": 1902.366
    },
    "500": {
      "recall": 0.8502729999999777,
      "mean_distances": 2290.5402
    },
    "600": {
      "recall": 0.8696729999999725,
      "mean_distances": 2669.4286
    },
    "700": {
      "recall": 0.8846949999999688,
      "mean_distances": 3040.4552
    },
    "800": {
      "recall": 0.8967209999999614,
      "mean_distances": 3404.7669
    },
    "900": {
      "recall": 0.906341999999956,
      "mean_distances": 3763.0798
    },
    "1000": {
      "recall": 0.9141899999999545,
      "mean_distances": 4115.7348
    }
  },
  "cheap_outdegree_conditional_6": {
    "100": {
      "recall": 0.5839579999999975,
      "mean_distances": 665.5113
    },
    "200": {
      "recall": 0.7131199999999923,
      "mean_distances": 1097.0123
    },
    "300": {
      "recall": 0.7783479999999873,
      "mean_distances": 1510.527
    },
    "400": {
      "recall": 0.8196109999999805,
      "mean_distances": 1912.2138
    },
    "500": {
      "recall": 0.8476729999999787,
      "mean_distances": 2303.5271
    },
    "600": {
      "recall": 0.8680189999999762,
      "mean_distances": 2685.8661
    },
    "700": {
      "recall": 0.8840389999999737,
      "mean_distances": 3059.9157
    },
    "800": {
      "recall": 0.8966939999999683,
      "mean_distances": 3428.1463
    },
    "900": {
      "recall": 0.9068619999999632,
      "mean_distances": 3789.857
    },
    "1000": {
      "recall": 0.9152809999999542,
      "mean_distances": 4146.584
    }
  },
  "arya_mount_diskann_inverse": {
    "100": {
      "recall": 0.4387970000000012,
      "mean_distances": 570.1429
    },
    "200": {
      "recall": 0.5587129999999996,
      "mean_distances": 923.922
    },
    "300": {
      "recall": 0.6275949999999975,
      "mean_distances": 1263.9761
    },
    "400": {
      "recall": 0.6737679999999922,
      "mean_distances": 1595.7878
    },
    "500": {
      "recall": 0.7076949999999923,
      "mean_distances": 1920.5233
    },
    "600": {
      "recall": 0.7332829999999931,
      "mean_distances": 2238.5406
    },
    "700": {
      "recall": 0.7545319999999941,
      "mean_distances": 2551.8247
    },
    "800": {
      "recall": 0.7720799999999907,
      "mean_distances": 2860.4392
    },
    "900": {
      "recall": 0.7866149999999865,
      "mean_distances": 3165.1071
    },
    "1000": {
      "recall": 0.7990079999999877,
      "mean_distances": 3466.3922
    }
  },
  "arya_mount_random_on_rejects_1p": {
    "100": {
      "recall": 0.5910039999999996,
      "mean_distances": 654.6706
    },
    "200": {
      "recall": 0.7148059999999935,
      "mean_distances": 1077.0428
    },
    "300": {
      "recall": 0.7775129999999877,
      "mean_distances": 1481.9196
    },
    "400": {
      "recall": 0.8159759999999842,
      "mean_distances": 1872.731
    },
    "500": {
      "recall": 0.842842999999976,
      "mean_distances": 2252.9372
    },
    "600": {
      "recall": 0.8623909999999716,
      "mean_distances": 2624.1443
    },
    "700": {
      "recall": 0.8773519999999684,
      "mean_distances": 2986.9615
    },
    "800": {
      "recall": 0.8893349999999669,
      "mean_distances": 3343.2492
    },
    "900": {
      "recall": 0.8989459999999639,
      "mean_distances": 3693.3817
    },
    "1000": {
      "recall": 0.9069019999999626,
      "mean_distances": 4037.8746
    }
  },
  "arya_mount_sigmoid_on_rejects_0p1": {
    "100": {
      "recall": 0.558856,
      "mean_distances": 636.969
    },
    "200": {
      "recall": 0.6852449999999922,
      "mean_distances": 1040.6039
    },
    "300": {
      "recall": 0.7509949999999921,
      "mean_distances": 1427.6492
    },
    "400": {
      "recall": 0.7922729999999889,
      "mean_distances": 1802.6884
    },
    "500": {
      "recall": 0.821336999999983,
      "mean_distances": 2168.0272
    },
    "600": {
      "recall": 0.8427739999999799,
      "mean_distances": 2524.7254
    },
    "700": {
      "recall": 0.859486999999977,
      "mean_distances": 2874.2433
    },
    "800": {
      "recall": 0.8727319999999738,
      "mean_distances": 3217.6562
    },
    "900": {
      "recall": 0.883475999999971,
      "mean_distances": 3555.4206
    },
    "1000": {
      "recall": 0.8926219999999657,
      "mean_distances": 3888.6658
    }
  },
  "arya_mount_sigmoid_on_rejects_1": {
    "100": {
      "recall": 0.5680900000000011,
      "mean_distances": 642.2113
    },
    "200": {
      "recall": 0.693724999999997,
      "mean_distances": 1049.4137
    },
    "300": {
      "recall": 0.759203999999987,
      "mean_distances": 1440.2528
    },
    "400": {
      "recall": 0.7997789999999848,
      "mean_distances": 1818.9091
    },
    "500": {
      "recall": 0.8280559999999785,
      "mean_distances": 2187.9423
    },
    "600": {
      "recall": 0.8492099999999757,
      "mean_distances": 2548.2357
    },
    "700": {
      "recall": 0.865243999999972,
      "mean_distances": 2901.1573
    },
    "800": {
      "recall": 0.8783309999999689,
      "mean_distances": 3247.6966
    },
    "900": {
      "recall": 0.8887979999999619,
      "mean_distances": 3588.6021
    },
    "1000": {
      "recall": 0.897569999999962,
      "mean_distances": 3924.4477
    }
  },
  "neighborhood_overlap": {
    "100": {
      "recall": 0.5924699999999973,
      "mean_distances": 653.0375
    },
    "200": {
      "recall": 0.7164259999999922,
      "mean_distances": 1074.6954
    },
    "300": {
      "recall": 0.7791309999999871,
      "mean_distances": 1479.3042
    },
    "400": {
      "recall": 0.8178389999999812,
      "mean_distances": 1869.9409
    },
    "500": {
      "recall": 0.8438809999999759,
      "mean_distances": 2249.6166
    },
    "600": {
      "recall": 0.8629399999999723,
      "mean_distances": 2619.7752
    },
    "700": {
      "recall": 0.8778459999999692,
      "mean_distances": 2981.8076
    },
    "800": {
      "recall": 0.8896199999999679,
      "mean_distances": 3337.4188
    },
    "900": {
      "recall": 0.899142999999966,
      "mean_distances": 3686.619
    },
    "1000": {
      "recall": 0.9070329999999627,
      "mean_distances": 4030.6445
    }
  },
  "sigmoid_ratio_1": {
    "100": {
      "recall": 0.43288199999999827,
      "mean_distances": 600.183
    },
    "200": {
      "recall": 0.5586499999999993,
      "mean_distances": 970.9547
    },
    "300": {
      "recall": 0.6310479999999988,
      "mean_distances": 1329.2605
    },
    "400": {
      "recall": 0.6798539999999964,
      "mean_distances": 1680.0179
    },
    "500": {
      "recall": 0.716152999999996,
      "mean_distances": 2023.4892
    },
    "600": {
      "recall": 0.7438769999999943,
      "mean_distances": 2361.0644
    },
    "700": {
      "recall": 0.766243999999991,
      "mean_distances": 2693.4557
    },
    "800": {
      "recall": 0.7847809999999921,
      "mean_distances": 3022.1084
    },
    "900": {
      "recall": 0.8004469999999901,
      "mean_distances": 3346.5714
    },
    "1000": {
      "recall": 0.8135819999999875,
      "mean_distances": 3666.4775
    }
  },
  "arya_mount_diskann": {
    "100": {
      "recall": 0.5670899999999996,
      "mean_distances": 636.1717
    },
    "200": {
      "recall": 0.6868309999999891,
      "mean_distances": 1037.2736
    },
    "300": {
      "recall": 0.7484879999999875,
      "mean_distances": 1420.4605
    },
    "400": {
      "recall": 0.787533999999984,
      "mean_distances": 1790.6546
    },
    "500": {
      "recall": 0.8148569999999854,
      "mean_distances": 2150.6472
    },
    "600": {
      "recall": 0.8352079999999789,
      "mean_distances": 2502.2605
    },
    "700": {
      "recall": 0.8511249999999778,
      "mean_distances": 2845.8678
    },
    "800": {
      "recall": 0.8639889999999736,
      "mean_distances": 3183.038
    },
    "900": {
      "recall": 0.8745399999999692,
      "mean_distances": 3515.0317
    },
    "1000": {
      "recall": 0.8834149999999632,
      "mean_distances": 3840.9451
    }
  },
  "nearest_m": {
    "100": {
      "recall": 0.4585029999999978,
      "mean_distances": 582.9379
    },
    "200": {
      "recall": 0.5710429999999981,
      "mean_distances": 932.192
    },
    "300": {
      "recall": 0.6332269999999977,
      "mean_distances": 1266.7815
    },
    "400": {
      "recall": 0.6740259999999968,
      "mean_distances": 1590.4145
    },
    "500": {
      "recall": 0.704502999999995,
      "mean_distances": 1905.7237
    },
    "600": {
      "recall": 0.7277089999999932,
      "mean_distances": 2215.0362
    },
    "700": {
      "recall": 0.746436999999995,
      "mean_distances": 2517.9533
    },
    "800": {
      "recall": 0.761753999999995,
      "mean_distances": 2815.3439
    },
    "900": {
      "recall": 0.7747229999999924,
      "mean_distances": 3108.6629
    },
    "1000": {
      "recall": 0.7862079999999941,
      "mean_distances": 3398.9792
    }
  },
  "furthest_m": {
    "100": {
      "recall": 0.06784699999999672,
      "mean_distances": 822.4317
    },
    "200": {
      "recall": 0.1133889999999959,
      "mean_distances": 1371.2665
    },
    "300": {
      "recall": 0.14857699999999616,
      "mean_distances": 1876.7751
    },
    "400": {
      "recall": 0.17785799999999719,
      "mean_distances": 2355.0474
    },
    "500": {
      "recall": 0.20298499999999953,
      "mean_distances": 2821.3461
    },
    "600": {
      "recall": 0.2256900000000007,
      "mean_distances": 3279.5219
    },
    "700": {
      "recall": 0.2458640000000004,
      "mean_distances": 3729.5916
    },
    "800": {
      "recall": 0.2645080000000004,
      "mean_distances": 4173.7489
    },
    "900": {
      "recall": 0.2815950000000004,
      "mean_distances": 4612.0338
    },
    "1000": {
      "recall": 0.29723899999999964,
      "mean_distances": 5044.1106
    }
  }
}
# --- End of JSON data ---

# Create the plot
fig, ax = plt.subplots(figsize=(14, 9)) # Adjusted figure size for better legend spacing

# Get a colormap
num_lines = len(data)
colors = cm.get_cmap('tab20', num_lines) # Using 'tab20' colormap for more distinct colors

# Plot each method
for i, (method_name, method_data) in enumerate(data.items()):
    # Sort keys numerically ('100', '200', ...)
    sorted_keys = sorted(method_data.keys(), key=int)

    recalls = [method_data[key]['recall'] for key in sorted_keys]
    mean_dists = [method_data[key]['mean_distances'] for key in sorted_keys]

    # Plot the data for this method
    ax.plot(recalls, mean_dists, marker='x', linestyle='-', label=method_name, color=colors(i))

# Set plot labels and title
ax.set_xlabel('Mean Recall')
ax.set_ylabel('Mean Distances (log scale)')
ax.set_title('Mean Distances vs. Mean Recall (SIFT Dataset)')

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Add grid
ax.grid(True, which='both', linestyle='--', linewidth=0.5) # 'both' for major and minor ticks on log scale

# Add legend
# Place legend outside the plot area to avoid overlap
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize='small')

# Adjust layout to prevent labels from being cut off
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for the external legend

# Save the plot
plt.savefig('mean_distances_vs_mean_recall.png', dpi=300, bbox_inches='tight')