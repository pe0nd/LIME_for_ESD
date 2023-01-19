import ModelHeat as mh
import pandas as pd
from itertools import product
import csv
import math

# Output file name
file_name = "<output_file_name>.csv"

pv_curve = ["real"]  # can be "uniform","real" or "parabola"
cloud_dist = ["random_hour"]  # can be "equal" or "random_hour"
dem_shape = ["uniform"]  # can be "uniform" or "real"
dem_heat_shape = ["uniform"]  # can be "uniform" or "real"
cloud_shape = ["dist"]  # can be "uniform" or "dist"
morning_mist = range(0, 13, 6)  # determines the time steps fow which morning mist is present
cloud_loss = range(4, 7)  # will later be multiplied by 0.1 to get loss in kWh
battery_cost = [540, 600, 660]
pv_surplus = range(8, 11)
cloud_num = range(4, 7)
hs_cost = [0]  # [45, 50, 55]  # change if heat should be considered
rnd = [1] * 5  # make this a list with n elements, n being the number of runs
isHeat = [False]  # if false the heat demand will be set to 0

# We define samples as a grid over all possible combinations. A sparse selection would also be possible.
sample_input = pd.DataFrame(list(product(pv_curve, pv_surplus, cloud_dist, cloud_loss, cloud_num, cloud_shape,
                                         dem_shape, battery_cost,
                                         dem_heat_shape, hs_cost, isHeat,
                                         rnd, morning_mist)),
                            columns=["pv_curve", "pv_surplus", "cloud_dist", "cloud_loss", "cloud_num", "cloud_shape",
                                     "dem_shape", "battery_cost",
                                     "dem_heat_shape", "hs_cost", "isHeat",
                                     "rnd", "morning_mist"])

# We define the index of the point of interest in sample_input
base_index = math.ceil(sample_input.__len__() / 2) - 1


def get_settings_for_sample(sample, lifetime=3650):
    """Creates the settings dict for a given sample"""
    # calculate base scenario as point of reference for distance calculation
    settings = mh.getSettings()
    settings["pv_curve"] = sample["pv_curve"]
    settings["pow_pv_surplus"] = sample["pv_surplus"]
    settings["cloud_dist"] = sample["cloud_dist"]
    settings["cloud_shape"] = sample["cloud_shape"]
    settings["pow_clouds"] = sample["cloud_loss"] * 0.1
    if sample["cloud_shape"] == "dist":
        settings["pow_clouds"] = [sample["cloud_loss"] * 0.1, 0.1]
    settings["num_clouds"] = sample["cloud_num"]

    settings["dem_shape"] = sample["dem_shape"]
    # we assume a lifetime of 10 years and divide the investment cost equally
    settings["cost_Battery"] = sample["battery_cost"] / lifetime

    settings["dem_heat_shape"] = sample["dem_heat_shape"]
    settings["cost_HeatStorage"] = sample["hs_cost"] / lifetime

    settings["morning_mist"] = sample["morning_mist"]

    if not sample_input.iloc[base_index]["isHeat"]:
        settings["dem_heat"] = 0
        settings["cost_HeatStorage"] = 10000

    return settings


# run ESD model for the point of interest
HMHeat = mh.HouseModel(settings_dict=get_settings_for_sample(sample_input.iloc[base_index]))
R_base = HMHeat.build_and_run()[0]


# Uncomment below to get a plot of electricity production and demand
import matplotlib.pyplot as plt
import plot_lib as pl

fig = plt.figure(figsize=(12, 8), dpi=80)
fig.add_subplot(2, 1, 1)
fig.add_subplot(2, 1, 2)
ax = fig.axes
ax[0], ax[1] = pl.plot_production_and_demand(R_base, isHeat=isHeat, title="base model", axis=ax)
fig.show()


# solve the ESD model for each variation
solutions = {}
for index, row in sample_input.iterrows():
    HMHeat = mh.HouseModel(settings_dict=get_settings_for_sample(row))
    # Get result and save it together with the distance to the base model
    solutions[solutions.__len__()] = mh.getKPI(HMHeat.build_and_run()[0], basemodel=R_base)


# save solutions to csv
with open(file_name, 'w', newline='') as csvfile:
    fieldnames = ['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss', 'morning_mist', 'Battery', 'HeatStorage',
                  "d_b", "d_p", "d_h"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

    writer.writeheader()
    for i in solutions:
        writer.writerow(solutions[i])
