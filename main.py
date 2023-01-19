import csv
import pandas as pd
import K_Lasso as KL
import plot_lib as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import preprocessing
import math


def open_samples_from_file(path):
    """Load data from a csv file"""
    data = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            data[data.__len__()] = {"c_Bat": float(row["c_Bat"]), "c_HS": float(row["c_HS"]),
                                    "surplus": float(row["surplus"]),
                                    "clouds": float(row["clouds"]), "energy_loss": float(row["energy_loss"]),
                                    "morning_mist": float(row["morning_mist"]),
                                    "Battery": float(row["Battery"]), "HeatStorage": float(row["HeatStorage"]),
                                    "d_b": float(row["d_b"]), "d_p": float(row["d_p"]), "d_h": float(row["d_h"])}
    data = pd.DataFrame(data).transpose()
    # take average for experiments with randomness
    return data.groupby(["c_Bat", "surplus", "clouds", "energy_loss", "c_HS",
                         "morning_mist"
                         ], as_index=False).agg("mean")


# In order to generate samples, take a look at generate_samples.py

# Open data.
data = open_samples_from_file('<input_file_name>.csv')

# normalize data
scaler = preprocessing.MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Index of the data sample containing the point of interest (used for distance calculation)
idx = math.ceil(data.__len__()/2) - 1
# number of desired interpretable features in the explanation
k = 1

# perform the weighted regression
r = KL.k_lasso(normalized_data[['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss', 'morning_mist']],
               normalized_data[["d_b", "d_p", "d_h"]],
               normalized_data['Battery'],
               k,
               sample_idx=idx)

# print output to console
temp = r.argsort()
temp2 = ['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss', 'morning_mist']
for i in range(k):
    ordinal = lambda n: "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
    print("The "+str(ordinal(i+1))+" most important feature is: " + temp2[temp[temp.__len__()-1-i]])

# # create a scatter plot of the neighborhood of the base sample
# calculate distance to all samples to highlight close points
dist = KL.calc_dist(idx, normalized_data[["d_b", "d_p", "d_h"]])
fig = plt.figure(figsize=(11, 3.5), dpi=80)
for i in range(5):
    fig.add_subplot(1, 5, i + 1)
ax = fig.axes
ax = pl.plot_neighborhood(idx, data[['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss']], data['Battery'],
                          distance=dist, neighbors=-1, axis=ax,
                          label_names=[r"$p_b~[\frac{€}{kWh}]$", r"$p_{HS}$", r"$s_{PV}~[kWh]$", "number \n of clouds",
                                       r"cloud size $[kWh]$", r"$C_{b}~[kWh]$"])
# turn of y-axis of all axes except the left most, for a compact display
for i in range(1, 5):
    ax[i].get_yaxis().set_visible(False)

# put weights as title and place axis closer together
for i in range(5):
    ax[i].set(title='weight ' + str(round(r[i], 4)))
    ax[i].set_position([0.07 + i * 0.18, 0.2, 0.17, 0.6])

fig.suptitle('Sample neighborhoods and weights with k=' + str(k))
fig.show()
