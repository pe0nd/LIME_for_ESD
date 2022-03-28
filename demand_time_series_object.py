import csv
import numpy as np
import matplotlib.pyplot as plt
from time_series_object import time_series_object


class demand_time_series_object(time_series_object):
    """
    Class to create demand time series for a HouseModel.

    Args:
        shape(str): type of demand series to create
        delta_time(float): weight of each time step, e.g. 1 if there is one time step per hour,
                            1/6 if there is a time step for every 10 minutes.
        length(int): number of time steps in the resulting time series.
        absolute_demand(float): absolute demand value for each hour for shape="uniform".
        data_path(str): relative path to a .csv file with time series data for shape="real"
        day_selection(int or array): if set selected days will be selected from the time series,
                                        days are assumed to be rows in the original file
        time_step_selection(str or array): if set, only selected time steps will be considered
        day_type_selection(str): if set, the column will be selected as a day type ,i.e. an encoding if working day
                                or weekend
    """
    def __init__(self, shape="uniform", delta_time=1, length=8760, absolute_demand=1,
                 data_path="timeseries/e_dem.csv", day_selection=None, time_step_selection=None,
                 day_type_selection=None):
        self.day_type = 1
        if shape == "uniform":  # build a uniform shape
            time_series_object.__init__(self,[absolute_demand * delta_time] * length)
        if shape == "real":  # build a demand series from real data
            ts = {}
            with open(data_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=';')
                for row in reader:
                    ts[ts.__len__()] = row

            # select day (rows)
            if day_selection is not None:
                t = ts[day_selection]
            else:
                t = ts

            if day_type_selection is not None:
                self.day_type = float(t.get(day_type_selection))

            # select time steps (columns)
            if time_step_selection is not None:
                t2 = {}
                for key in time_step_selection:
                    t2[key] = t.get(key)
                t = t2

            # ensure data type and transform into array
            t1 = []
            for i in t.values():
                t1.append(float(i))

            # scale time series to fit demand
            st = delta_time * 24 * absolute_demand / sum(t1)
            t1 = np.multiply(t1, st)
            # fill in missing time steps to ensure that the final time series has time_step_length
            time_series_object.__init__(self, np.repeat(t1, length/t1.__len__()))

    def get_day_type(self):
        return self.day_type
