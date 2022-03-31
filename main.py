import csv
import pandas as pd
import K_Lasso as KL
import plot_lib as pl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Open data.
# In order to generate samples have a look at generate_samples
def open_samples_from_file(path):
    data = {}
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            data[data.__len__()] = {"c_Bat": float(row["c_Bat"]), "c_HS": float(row["c_HS"]),
                                    "surplus": float(row["surplus"]),
                                    "clouds": float(row["clouds"]), "energy_loss": float(row["energy_loss"]),
                                    "Battery": float(row["Battery"]), "HeatStorage": float(row["HeatStorage"]),
                                    "d_b": float(row["d_b"]), "d_p": float(row["d_p"]), "d_h": float(row["d_h"])}
    return pd.DataFrame(data).transpose()


data = open_samples_from_file('Sample_run.csv')

# perform LIME
k = 1
r = KL.k_lasso(data[['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss']], data[["d_b", "d_p", "d_h"]],
               data['Battery'], k, sample_idx=121)

# create a scatter plot of the neighborhood of the base sample
dist = KL.calc_dist(121, data[["d_b", "d_p", "d_h"]])
fig = plt.figure(figsize=(11, 3.5), dpi=80)
for i in range(5):
    fig.add_subplot(1, 5, i + 1)
ax = fig.axes
ax = pl.plot_neighborhood(121, data[['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss']], data['Battery'],
                          distance=dist, neighbors=-1, axis=ax,
                          label_names=[r"$p_b~[\frac{€}{kWh}]$", r"$p_{HS}$", r"$s_{PV}~[kWh]$", "number \n of clouds",
                                       r"cloud size $[kWh]$", r"$C_{b}~[kWh]$"])
# turn of y-axis of all axes except the left most, for a compact display
for i in range(1, 5):
    ax[i].get_yaxis().set_visible(False)

# put lime weights as title and place axis closer together
for i in range(5):
    ax[i].set(title='weight ' + str(round(r[i], 4)))
    ax[i].set_position([0.07 + i * 0.18, 0.2, 0.17, 0.6])

fig.suptitle('Sample neighborhoods and weights of LIME with K=' + str(k))
fig.show()

# # compare different results
# define files, scenario names and colors
files = ["Sample_run.csv", "Sample_run.csv"]
scenario_names = ["Scenario1", "Scenario2"]
color = ["blue", "orange"]

# apply LIME on each scenario
weights = []
for file in files:
    temp = open_samples_from_file(file)
    weights.append(KL.k_lasso(temp[['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss']], temp[["d_b", "d_p", "d_h"]],
                              temp['Battery'], k, sample_idx=121))

# plot results as bars
fig = plt.figure(figsize=(11, 3.5), dpi=80)
for i in range(5):
    fig.add_subplot(1, 5, i + 1)
ax = fig.axes
data = pd.DataFrame(weights)
data = data.rename(columns={0: r"$p_b~[\frac{€}{kWh}]$", 1: r"$p_{HS}$", 2: r"$s_{PV}~[kWh]$",
                            3: "number \n of clouds", 4: r"cloud size $[kWh]$"})

ax = pl.plot_bars(data, col=color, axis=ax)

# turn of y-axis of all axes except the left most, for a compact display
for i in range(1, 5):
    ax[i].get_yaxis().set_visible(False)

# place axis closer together
for i in range(5):
    ax[i].set_position([0.07 + i * 0.18, 0.2, 0.17, 0.6])
fig.suptitle('Scenario comparison')

# add a legend
legend_elements = []
for scen in range(scenario_names.__len__()):
    legend_elements.append(Line2D([0], [0], color=color[scen], lw=4, label=scenario_names[scen]))
fig.legend(handles=legend_elements, loc=8, borderaxespad=0., prop={'size': 12}, ncol=2)

fig.show()
