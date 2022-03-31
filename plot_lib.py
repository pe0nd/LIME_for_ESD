import warnings

import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


def plot_production_and_demand(model_result, isHeat=False, title="Model 1", d_t=1 / 6, show_storage=False, axis=None):
    """
    Plot production of electricity (and heat) on positive y-axis and consumption on negative axis.

    Args:
        model_result(ConcreteModel): A solved instance of the ModelHeat class
        isHeat(bool): if True, heat sector is plotted as the second output. Default is False
        title(str): model name to be used in the axis title
        d_t(float): delta time for correct time scaling of the x-axis
        show_storage(bool): if True, storage level is plotted on the second y-axis
        axis(matplotlib.axis): existing matplotlib axis where plot will be attached,
            if None is given a new axis will be created

    :return:
        ax(matplotlib.axis): axis with electricity production and demand plot
        axh(matplotlib.axis): axis with heat production and demand plot
    """
    col_e = np.array(["navy", "purple", "blue", "forestgreen", "navy", "orange"])
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
        ax.stackplot(x, y[[2, 0], :], labels=[r'$D_e(t)$', r'$E_b^{in}(t)$'], baseline='zero',
                     colors=list(col_e[[2, 0]]))
    ax.set(ylabel="kW", xticks=np.arange(0, 25, 4), xlabel="h",
           title="Electricity production and demand of " + title)
    if show_storage:
        ax2 = ax.twinx()
        z = np.array([pyo.value(model_result.EnergyBattery[i]) for i in range(x.__len__())])
        ax2.plot(x, z, label='Bat_level', color="black")
        ax2.set(ylabel="kWh")
        y_abs = max([pyo.value(model_result.EnergyBattery[i]) for i in range(x.__len__())])
        ax2.set_ylim([-y_abs, y_abs])
    ax.legend(bbox_to_anchor=(.0, .5), loc=6, prop={'size': 12})

    if isHeat:
        col_h = np.array(["red", "purple", "darkred", "darkred"])
        y = np.array([-pyo.value(model_result.DemandHeat[i]) for i in range(x.__len__())])
        y = np.vstack([y, [pyo.value(model_result.HeatPump_Heat[i]) for i in range(x.__len__())]])
        y = np.vstack([y, [-pyo.value(model_result.EnergyHeatStorage_IN[i]) for i in range(x.__len__())]])
        y = np.vstack([y, [pyo.value(model_result.EnergyHeatStorage_OUT[i]) for i in range(x.__len__())]])
        axh.stackplot(x, y[[3, 1], :], labels=[r'$E_{HP}^{out}(t)$', r'$E_{HS}^{in}(t)$'], baseline='zero',
                      colors=col_h[[3, 1]])
        axh.stackplot(x, y[[0, 2], :], labels=[r'$D_h(t)$', r'$E_{HS}^{in}(t)$'], baseline='zero', colors=col_h[[0, 2]])
        axh.set(ylabel="kW", xticks=np.arange(0, 25, 4), xlabel="h",
                title="Heat production and demand of " + title)
        if show_storage:
            axh2 = axh.twinx()
            z = np.array([pyo.value(model_result.EnergyHeatStorage[i]) for i in range(144)])
            axh2.plot(x, z, label='P_HS', color="black")
            axh2.set(ylabel="kWh")
            y_abs = max([pyo.value(model_result.EnergyHeatStorage[i]) for i in range(x.__len__())])
            axh2.set_ylim([-y_abs, y_abs])
        axh.legend(bbox_to_anchor=(.0, .5), loc=6, prop={'size': 12})

    return ax, axh


def plot_production_time_series(model_result, title="Model 1", d_t=1 / 6, axis=None):
    """
        Plot PV production, electricity demand and storage level.

        Args:
            model_result(ConcreteModel): A solved instance of the ModelHeat class
            title(str): model name to be used in the axis title
            d_t(float): delta time for correct time scaling of the x-axis
            axis(matplotlib.axes): existing matplotlib axis where plot will be attached,
                if None is given a new axis will be created

        :return:
            ax(matplotlib.axis): axis with PV production and demand plot
        """
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
    return ax


def smooth_step(x, l1, l2):
    """smooth step function to get a smooth step between 0 and 1"""
    if x <= l1:
        r = 0
    elif l1 < x <= l2:
        r = 3 * pow(x, 2) - 2 * pow(x, 3)
    else:
        r = 1
    return r


def plot_neighborhood(base_idx, samples, target, distance=None, neighbors=-1, label_names=None, axis=None):
    """
    Scatter plot  a neighborhood of a sample.

    Args:
        base_idx(int): Index of the base sample in "samples"
        samples(pandas.DataFrame): DataFrame of all samples with features in the columns
        target(pandas.DataFrame): DataFrame of the target values corresponding to the samples
        distance(list): list of distances
        neighbors(int): number of closest neighbors to highlight.
            If 0 or negative all neighbors will be only highlighted based on their distance.
        label_names(list): list with alternative labels for the plots.
            Length have to be number of columns in samples +1, the last entry is the alternative y-label for target.
            If None is given, column names will be used as labels.
        axis(matplotlib.axes): existing matplotlib axis where plot will be attached,
            if None is given a new axis will be created

    :return:
        ax(matplotlib.axis): axis with PV production and demand plot, list of axis if samples has multiple columns
    """
    if axis is None:
        ax = []
    else:
        ax = axis
    if distance is None:
        distance = [1]*target.__len__()

    # normalize the distances and apply the smooth step
    distance = pd.DataFrame(distance)
    distance = distance[0].sort_values(ascending=False)
    distance = distance[1:target.__len__()]
    scaler = preprocessing.MinMaxScaler()
    area = pd.DataFrame(scaler.fit_transform(pd.DataFrame(distance)))
    if neighbors > 0:
        area = area.applymap(lambda a: smooth_step(a, distance[neighbors], 1))
    area = area.clip(upper=9.99e-1, lower=0.1)  # limits to show all points at least a little
    color = ["blue"]



    for i in range(samples.columns.__len__()):
        if axis is None:
            ax.append(plt.axes())
        # add scatter points
        ax[i].scatter(samples[samples.columns[i]][distance.index], target[distance.index],
                      s=30 * (area + .1), alpha=area, c=color)
        # add x for base sample
        ax[i].scatter(samples[samples.columns[i]][base_idx], target[base_idx], s=30,
                      c="red", marker='x')
        if label_names is None:
            ax[i].set(xlabel=samples.columns[i], ylabel=target.columns[0])
        else:
            ax[i].set(xlabel=label_names[i], ylabel=label_names[-1])

    return ax


def plot_bars(data, label_names=None, col="blue", axis=None):
    """
    Bar plot of different LIME or LASSO results.

    Args:
        data(pandas.DataFrame): DataFrame with features as columns and different results as rows
        label_names(list): list with alternative feature names.
            If None, column names are used
        axis(matplotlib.axis): existing matplotlib axis where plot will be attached,
            if None is given a new axis will be created

    :return:
        ax(matplotlib.axis): axis with bar plot, list of axis if data has multiple columns
    """
    if axis is None:
        ax = []
    else:
        ax = axis

    min_v = min(data.min(axis=1))
    max_v = max(data.max(axis=1))
    for i in range(data.columns.__len__()):
        if axis is None:
            ax.append(plt.axes())
        for idx in range(data.index.__len__()):
            ax[i].bar(idx, data[data.columns[i]][idx], fill=col[idx])
        ax[i].set_ylim([min_v, max_v])
        if label_names is None:
            ax[i].set(title=data.columns[i], ylabel='weight')
        else:
            ax[i].set(title=label_names[i], ylabel='weight')
        ax[i].get_xaxis().set_visible(False)

    return ax