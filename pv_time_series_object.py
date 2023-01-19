import csv
import numpy as np
import matplotlib.pyplot as plt
from time_series_object import time_series_object
import sympy
import math
import warnings
import random


class pv_time_series_object(time_series_object):
    """
    Class to create demand time series for a HouseModel.

    Args:
        shape(str): type of demand series to create
        delta_time(float): weight of each time step, e.g. 1 if there is one time step per hour,
                            1/6 if there is a time step for every 10 minutes.
        length(int): number of time steps in the resulting time series.
        surplus(float): surplus a timeseries should have
        demand_ts(array): array of demand values, has to be same size as length
        data_path(str): relative path to a .csv file with time series data for shape="real"
        day_selection(int or array): if set selected days will be selected from the time series,
                                        days are assumed to be rows in the original file
        clouds(int): number of clouds that should be mapped to the pv time series
        cloud_size(float or array): size of a single cloud in kWh, array with estimate and variance for random clouds
        cloud_dist(str): distribution of the clouds
        cloud_shape(str): shape of the individual clouds
        pv_limit(array of size 2): limits where clouds are put in between, for shape=="uniform" also sunrise and sunset
    """

    def __init__(self, shape="uniform", delta_time=1 / 6, length=144, surplus=1, demand_ts=None,
                 data_path="timeseries/PV_TS.csv", day_selection=None, clouds=0, cloud_size=0, cloud_dist="equal",
                 cloud_shape="pow", rescale=True, pv_limit=None, morning_mist=0):

        if pv_limit is None:
            pv_limit = [42, 126]
        if demand_ts is None:
            demand_ts = [1 / 6] * 144

        ts = [0] * length

        # # # "Find basic structure of PV shape" # # #

        # constant pv shape above the demand starting at pv_limit[0] ending at pv_limit[1]
        if shape == "box" or shape == "uniform":

            # single box as pv
            for i in range(length):
                if pv_limit[0] <= i < pv_limit[1]:
                    ts[i] = demand_ts[i] + (surplus / (pv_limit[1] - pv_limit[0]))

        # a 3-step function for the pv curve
        if shape == "3box":
            # 3 boxes as pv curve
            pv_limit = [7, 21, 9, 19, 11, 17]

            for i in range(length):
                if pv_limit[0] <= i < pv_limit[1]:
                    ts[i] = .5 * demand_ts[i]
                if pv_limit[2] <= i < pv_limit[3]:
                    ts[i] = 1 * demand_ts[i]
                if pv_limit[4] <= i < pv_limit[5]:
                    ts[i] = demand_ts[i] + (surplus / (pv_limit[4] - pv_limit[5]))

        # PV curve has a parabola shape, Only tested for constant demand
        if shape == "par" or shape == "parabola":
            dem = sum(demand_ts[j] for j in range(int(1 / delta_time)))
            a = sympy.Symbol('x')
            z = sympy.Symbol('z')
            A = sympy.Symbol('A')
            l_h = [pv_limit[0] * delta_time, pv_limit[1] * delta_time]
            # formulating quadratic equation (parabola is shifted for easier calculation)
            f = -z * a * (a - l_h[1] + l_h[0]) - dem
            # f = -z * (a-l_h[0]) * (a - l_h[1]) - dem
            solution = sympy.solve(f, a)
            b = sympy.integrate(f, (a, solution[0], solution[1]))
            b = sympy.simplify(b)
            eq = sympy.Eq(A, b)
            c = sympy.solve(eq, z)
            ss = []
            for sol in c:
                ss.append(sympy.simplify(sol))

            # solve equation for z
            solu = []
            for i in ss:
                solu.append(i.subs(A, surplus).evalf())

            # solving quadratic equations can yield imaginary numbers and since solutions are numerical
            # they will most of the time. We chose the solution with the smallest imaginary part.
            def getImaginary(elem):
                return abs(sympy.im(elem))

            solu.sort(key=getImaginary)

            # z1 = sympy.solve(eq.subs(A, model.Settings["pow_pv_surplus"]))[0].evalf()
            z1 = float(sympy.re(solu[0]))
            # the warning below is for debugging purpose and will only be called if there was no solution with
            # sufficiently small imaginary part
            if abs(sympy.im(solu[0])) > 1e-4:
                print("Threw away large imaginary part of size " + str(sympy.im(solu[0])))
                print("Sample is: Surplus_" + str(surplus) + " Clouds_" +
                      str(clouds))

            for i in range(length):
                x = i * delta_time
                ts[i] = float(max(0, -z1 * (x * x - (l_h[0] + l_h[1]) * x + (l_h[0] * l_h[1]))) * delta_time)
                # ts[i] = float(max(0, -z1 * (x * x - (pv_limit[0] + pv_limit[1]) * x +
                #                             (pv_limit[0] * pv_limit[1]))) * delta_time)

            # t = sum(max(0, ts[i] - delta_time) - 1 for i in range(length)) * delta_time

        # data from a real dataset
        if shape == "real":
            t = {}
            with open(data_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',')
                for row in reader:
                    t[t.__len__()] = row

            # select day (rows)
            if day_selection is not None:
                t = t[day_selection]
                t1 = []
                for i in t.values():
                    t1.append(float(i) * delta_time)
            else:
                t1 = []
                for i in t.values():
                    for j in i.values():
                        t1.append(float(j) * delta_time)

            # Transform into right time step length (e.g., hourly data -> 15 min data)
            t1 = np.repeat(t1, int(length / t1.__len__()))

            # To find the right scaling factor interval doubling and interval halving is used
            sp = 0
            scaling_factor = .5
            while sp < surplus:
                scaling_factor = scaling_factor * 2
                sp = sum(max(0, scaling_factor * t1[i] - demand_ts[i]) for i in range(demand_ts.__len__()))
            # interval halving
            UB = scaling_factor
            LB = scaling_factor / 2
            for i in range(100):
                sp = sum(
                    max(0, (LB + 0.5 * (UB - LB)) * t1[i] - demand_ts[i]) for i in range(demand_ts.__len__()))
                if sp > surplus:
                    UB = (LB + 0.5 * (UB - LB))
                else:
                    LB = (LB + 0.5 * (UB - LB))
                if abs(LB - UB) < 1e-4:
                    break
            # apply rescale
            for i in range(length):
                ts[i] = UB * t1[i]

            # # find limits for cloud placement where av_PV > Demand
            # t = []
            # for l1, l2 in zip(ts, demand_ts):
            #     t.append(l1 - l2)
            # t = np.array(t)
            # pv_limit = [np.where(t > 0)[0][0], 24/delta_time - np.where(t[::-1] > 0)[0][0]]

            # find sunrise and sunset
            t = np.array(ts)
            pv_limit = [np.where(t > 0)[0][0], 24/delta_time - np.where(t[::-1] > 0)[0][0]]

        # # # "Morning mist reduces the first timesteps of the pv time series" # # #
        if morning_mist > 0:
            # get av_PV >0
            t = np.array(ts)
            sunrise = np.where(t > 0)[0][0]
            for i in range(0, morning_mist):
                ts[sunrise+i] = 0
            # updating production start for correct surplus calculation
            pv_limit[0]= max(sunrise+morning_mist, pv_limit[0])


        # # # "Find cloud starting points from distribution" # # #

        if clouds > 0:
            dips_idx = []
            # clouds start at random starting points (full hour)
            if cloud_dist == "random_hour":  # random clouds, starting on full hours
                # random dips created by clouds
                random.Random(1000)
                if clouds > (pv_limit[1] - pv_limit[0]):
                    warnings.warn(
                        "There are " + str(pv_limit[1] - pv_limit[0]) + " possible starting points for clouds in " +
                        "this PV time series but " + str(clouds) + " clouds to be placed.")
                # it can happen that sample throws an error if there are not enough samples to draw :
                # e.g. draw 3 samples from [1,2]
                dips_idx = random.sample(
                    range(int(pv_limit[0]), int(pv_limit[1]), int(1 / delta_time)),
                    clouds)

            # clouds have an equal distance (rounded to full time steps)
            if cloud_dist == "equal":
                # get approximate midpoint of clouds
                if isinstance(cloud_size, list):  # this argument will be a list for cloud_shape == "dist"
                    p_c = cloud_size[0]
                else:
                    p_c = cloud_size
                approx_cloud_length = math.floor(math.ceil(p_c / delta_time) / 2)
                for i in range(clouds):
                    dips_idx.append(int(pv_limit[0] + ((i + 1) / (1 + clouds)) *
                                        (pv_limit[1] - pv_limit[0])) - approx_cloud_length)

            # # # "Place Clouds according to shape" # # #

            # all clouds lose an equal amount of energy
            if cloud_shape == "pow" or cloud_shape == "uniform":
                pow_redu = cloud_size
                demand_loss = dips_idx.__len__() * cloud_size
                idx_offset = 0
                while pow_redu > 0:
                    for i in dips_idx:
                        ts[i + idx_offset] = max(0, demand_ts[i + idx_offset] - pow_redu)
                    pow_redu -= min(demand_ts[i + idx_offset],
                                    pow_redu)
                    idx_offset += 1


                # all clouds have a fixed length, with cloud_size as number of time steps
            if cloud_shape == "fixed":
                pow_redu = cloud_size
                demand_loss = 0
                idx_offset = 0
                while pow_redu > 0:
                    for i in dips_idx:
                        demand_loss += demand_ts[i + idx_offset]
                        ts[i + idx_offset] = 0
                    pow_redu -= 1
                    idx_offset += 1

            # clouds are random from a normal distribution
            if cloud_shape == "dist":
                pow_dist = cloud_size
                demand_loss = 0

                for i in dips_idx:  # calculate individual cloud
                    pow_redu = max(random.gauss(pow_dist[0], pow_dist[1]), 0)
                    idx_offset = 0
                    while pow_redu > 0:
                        demand_loss += min(demand_ts[i + idx_offset], pow_redu)
                        ts[i + idx_offset] = max(0, demand_ts[i + idx_offset] - pow_redu)
                        pow_redu -= demand_ts[i + idx_offset]
                        idx_offset += 1

        # rescale the pv time series to fit set surplus
        if rescale:
            total_energy_is = 0
            wanted_surplus = surplus + demand_loss  # surplus now takes also the loss during the day into account
            num_steps = 0
            for i in range(ts.__len__()):
                total_energy_is += max(0, ts[i] - demand_ts[i])
                if ts[i] > demand_ts[i]:
                    num_steps += 1
            adj_factor = (wanted_surplus - total_energy_is) / num_steps

            for i in range(ts.__len__()):
                if ts[i] > demand_ts[i]:
                    ts[i] += adj_factor

        # import matplotlib.pyplot as plt
        # plt.plot(ts)
        # plt.show()
        # print(ts)
        time_series_object.__init__(self, ts)


if __name__ == '__main__':
    from demand_time_series_object import demand_time_series_object as dso

    time_step_selection = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                           "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                           "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                           "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
    d_t = dso(shape="uniform",
              absolute_demand=1,
              delta_time=1 / 6,
              length=144,
              data_path="timeseries/e_dem.csv",
              day_selection=140,
              time_step_selection=time_step_selection,
              day_type_selection="Day")

    t = pv_time_series_object(shape="real", delta_time=1 / 6, length=144, surplus=4, demand_ts=d_t,
                              data_path="timeseries/PV_TS.csv", day_selection=140, clouds=2, cloud_size=0.5,
                              cloud_dist="equal",
                              cloud_shape="fixed", pv_limit=None)

