import csv
import numpy as np
import matplotlib.pyplot as plt


class time_series_object:
    """
    Class to create demand time series for a HouseModel.

    Args:
        TS(array) time series to be stored in the object
    """
    def __init__(self, TS):
        self.TS = TS

    def __getitem__(self, item):
        return self.TS[item]

    def __len__(self):
        return self.TS.__len__()

    def get_array(self):
        return self.TS

    def get_plot(self):
        return plt.plot(self.TS)