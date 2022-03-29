import warnings

import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np


def plot_production_and_demand(model_result, isHeat=False, title="Model 1", d_t=1 / 6, show_storage=False, axis=None):
    # defining a color scheme for electricity

    col_e = np.array(["navy", "purple", "blue", "yellow", "navy", "orange"])
    x = np.arange(0, model_result.Demand.__len__() * d_t, d_t)
    if axis is None:
        ax = plt.axes()
        axh = None
    if isinstance(axis, list) and isHeat:
        if axis.__len__() < 2:
            warnings.warn("Axis needs at least 2 elements if model has heat.")
        axh = axis[1]
        ax = axis[0]
    if isinstance(axis, list) and not isHeat:
        axh = None
        ax = axis[0]
    if not isinstance(axis, list) and not isHeat:
        axh = None
        ax = axis

    # demand side
    y = np.array([-pyo.value(model_result.EnergyBattery_IN[i]) for i in range(x.__len__())])
    y = np.vstack([y, [-pyo.value(model_result.HeatPump_Electricity[i]) for i in range(x.__len__())]])
    y = np.vstack([y, [-pyo.value(model_result.Demand[i]) for i in range(x.__len__())]])
    # production side
    y = np.vstack([y, [pyo.value(model_result.EnergyPV[i]) for i in range(x.__len__())]])
    y = np.vstack([y, [pyo.value(model_result.EnergyBattery_OUT[i]) for i in range(x.__len__())]])
    y = np.vstack([y, [pyo.value(model_result.EnergyBuy[i]) for i in range(x.__len__())]])

    if isHeat:
        ax.stackplot(x, y[[4, 5, 3], :], labels=[r'$E_b^{out}(t)$', r'$E_e(t)$', r'$E_{PV}(t)$'], baseline='zero',
                     colors=list(col_e[[4, 5, 3]]))
        ax.stackplot(x, y[[2, 0, 1], :], labels=[r'$D_e(t)$', r'$E_b^{in}(t)$', r'$E_{HP}^{in}(t)$'], baseline='zero',
                     colors=list(col_e[[2, 0, 1]]))
    else:
        ax.stackplot(x, y[[4, 5, 3], :], labels=[r'$E_b^{out}(t)$', r'$E_e(t)$', r'$E_{PV}(t)$'], baseline='zero',
                     colors=list(col_e[[4, 5, 3]]))
        ax.stackplot(x, y[[2, 0], :], labels=[r'$D_e(t)$', r'$E_b^{in}(t)$'], baseline='zero', colors=list(col_e[[2, 0]]))
    ax.set(ylabel="kW", xticks=np.arange(0, 25, 4), xlabel="h",
           title="Electricity production and demand of " + title)
    if show_storage:
        ax2 = ax.twinx()
        z = np.array([pyo.value(model_result.EnergyBattery[i]) for i in range(x.__len__())])
        ax2.plot(x, z, label='Bat_level', color = "black")
        ax2.set(ylabel="kWh")
        y_abs =max([pyo.value(model_result.EnergyBattery[i]) for i in range(x.__len__())])
        ax2.set_ylim([-y_abs, y_abs])
    ax.legend(bbox_to_anchor=(.0, .5),loc=6, prop={'size': 12})

    if isHeat:
        col_h = np.array(["red", "purple", "darkred", "darkred"])
        y = np.array([-pyo.value(model_result.DemandHeat[i]) for i in range(x.__len__())])
        y = np.vstack([y, [pyo.value(model_result.HeatPump_Heat[i]) for i in range(x.__len__())]])
        y = np.vstack([y, [-pyo.value(model_result.EnergyHeatStorage_IN[i]) for i in range(x.__len__())]])
        y = np.vstack([y, [pyo.value(model_result.EnergyHeatStorage_OUT[i]) for i in range(x.__len__())]])
        axh.stackplot(x, y[[3, 1], :], labels=[r'$E_{HP}^{out}(t)$', r'$E_{HP}^{in}(t)$'], baseline='zero', colors=col_h[[3, 1]])
        axh.stackplot(x, y[[0, 2], :], labels=[r'$D_h(t)$', r'$E_{HP}^{in}(t)$'], baseline='zero', colors=col_h[[0, 2]])
        axh.set(ylabel="kW", xticks=np.arange(0, 25, 4), xlabel="h",
                title="Heat production and demand of " + title)
        if show_storage:
            axh2 = axh.twinx()
            z = np.array([pyo.value(model_result.EnergyHeatStorage[i]) for i in range(144)])
            axh2.plot(x, z, label='P_HS', color="black")
            axh2.set(ylabel="kWh")
            y_abs = max([pyo.value(model_result.EnergyHeatStorage[i]) for i in range(x.__len__())])
            axh2.set_ylim([-y_abs, y_abs])
        axh.legend(bbox_to_anchor=(.0, .5),loc=6, prop={'size': 12})

    return ax, axh


def plot_production_time_series(model_result, title="Model 1", d_t=1 / 6,  axis=None):
    x = np.arange(0, model_result.Demand.__len__() * d_t, d_t)
    if axis is None:
        ax = plt.axes()
    else:
        ax = axis

    y = np.array([pyo.value(model_result.Demand[i]) for i in range(x.__len__())])
    y = np.vstack([y, [pyo.value(model_result.EnergyPV[i]) for i in range(x.__len__())]])

    ax.plot(x, y.transpose())
    ax.set(ylabel="kW", xticks=np.arange(0, model_result.Demand.__len__() * d_t, 4), xlabel="h",
           title="PV time series, demand and \n battery usage of " + title)
    ax2 = ax.twinx()
    z = np.array([pyo.value(model_result.EnergyBattery[i]) for i in range(x.__len__())])
    ax2.plot(x, z)
    ax2.set(ylabel="kWh")


