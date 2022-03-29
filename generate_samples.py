import ModelHeat as mh
import pandas as pd
from itertools import product
import csv

pv_curve = ["uniform"]  # can also be "real" or "parabola"
cloud_dist = ["equal"]  # can also be "random_hour"
dem_shape = ["uniform"]  # can also be "real"
dem_heat_shape = ["uniform"]  # can also be "real"
cloud_loss = range(4, 7)  # will later be multiplied by 0.1 to get loss in kWh
battery_cost = [540, 600, 660]
pv_surplus = [14.5, 15, 15.5]
cloud_num = range(4, 7)
hs_cost = [40, 50, 60]  # change if heat is should be considered
rnd = [1]  # make this a list with n elements, n being the number of runs that should be done
isHeat = [True]  # if false the heat demand will be set to 0

# We define samples as a grid over all possible combinations. A sparse selection would also be possible.
sample_input = pd.DataFrame(list(product(pv_curve, pv_surplus, cloud_dist, cloud_loss, cloud_num,
                                         dem_shape, battery_cost,
                                         dem_heat_shape, hs_cost, isHeat,
                                         rnd)),
                            columns=["pv_curve", "pv_surplus", "cloud_dist", "cloud_loss", "cloud_num",
                                     "dem_shape", "battery_cost",
                                     "dem_heat_shape", "hs_cost", "isHeat",
                                     "rnd"])

base_index = 40  # this should be the index in sample_input of the base sample


def get_settings_for_sample(sample, lifetime=3650):
    """Creates the settings dict for a given sample"""
    # calculate base scenario as point of reference for distance calculation
    settings = mh.getSettings()
    settings["pv_curve"] = sample["pv_curve"]
    settings["pow_pv_surplus"] = sample["pv_surplus"]
    settings["cloud_dist"] = sample["cloud_dist"]
    settings["pow_clouds"] = sample["cloud_loss"]*0.1
    settings["num_clouds"] = sample["cloud_num"]

    settings["dem_shape"] = sample["dem_shape"]
    # we assume a lifetime of 10 years and devide the investment cost equally
    settings["cost_Battery"] = sample["battery_cost"]/lifetime

    settings["dem_heat_shape"] = sample["dem_heat_shape"]
    settings["cost_HeatStorage"] = sample["hs_cost"]/lifetime
    if not sample_input.iloc[base_index]["isHeat"]:
        settings["dem_heat"] = 0

    return settings


HMHeat = mh.HouseModel(settings_dict=get_settings_for_sample(sample_input.iloc[base_index]))
R_base = HMHeat.build_and_run()[0]

# # Uncomment below to get a plot of production and consumption
# import matplotlib.pyplot as plt
# import plot_lib as pl
# fig = plt.figure(figsize=(12,8), dpi=80)
# fig.add_subplot(2, 1, 1)
# fig.add_subplot(2, 1, 2)
# ax = fig.axes
# ax[0], ax[1] = pl.plot_production_and_demand(R_base, isHeat=True, title="base model", axis=ax)
# fig.show()


# once we have the base sample for reference we can solve the energy system model for all other samples
solutions = {}
for index, row in sample_input.iterrows():
    HMHeat = mh.HouseModel(settings_dict=get_settings_for_sample(row))
    # Get result and save it together with the distance to the base model
    solutions[solutions.__len__()] = mh.getKPI(HMHeat.build_and_run()[0], basemodel=R_base)

# save solutions to csv

with open('Sample_run.csv', 'w', newline='') as csvfile:
    fieldnames = ['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss', 'Battery', 'HeatStorage',
                  "d_b", "d_p", "d_h"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

    writer.writeheader()
    for i in solutions:
        writer.writerow(solutions[i])