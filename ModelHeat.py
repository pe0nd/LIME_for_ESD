# file  -- Module.py --
import demand_time_series_object as dtso
import pv_time_series_object as ptso
import pyomo.environ as pyo
import numpy as np
import random
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition


def getSettings():
    """Returns a dictionary with default settings for usage in an HouseModel class object"""
    settingsDict = {"time": 24 * 6,
                    "cost_Battery": 30 / 365,
                    "cost_buy": 0.25,
                    "dem": 1,
                    "num_clouds": 0,
                    "pow_clouds": 0,
                    "pow_pv_surplus": 5,
                    "cost_HeatStorage": 5 / 365,
                    "HP_COP": 3,
                    "dem_heat": 2,
                    "pv_curve": "box",
                    "cloud_dist": "equal",
                    "cloud_pow": "pow",
                    "dem_shape": "uniform",
                    "dem_heat_shape": "uniform",
                    "non_reductive": True
                    }
    return settingsDict


class HouseModel:
    """Create an instance of the House model, setting its basic parameters"""

    def __init__(self, settings_dict=None):
        if settings_dict is None:
            self.Settings = getSettings()
        else:
            self.Settings = settings_dict
        self.pv_curve = settings_dict["pv_curve"]
        self.cloud_dist = settings_dict["cloud_dist"]
        self.cloud_pow = settings_dict["cloud_pow"]
        self.dem_shape = settings_dict["dem_shape"]
        self.dem_heat_shape = settings_dict["dem_heat_shape"]
        self.non_reductive = settings_dict["non_reductive"]
        self.model = pyo.ConcreteModel()


    def build(self):
        """Initiate the model and run it."""
        # Create an instance of the model
        model = pyo.ConcreteModel()

        # Define index sets and preprocess inputs
        time = range(self.Settings["time"])
        delta_time = 1 / (self.Settings["time"] / 24)

        # electric parameters
        cost_Battery = self.Settings["cost_Battery"]  # *((time[-1]+1)/8760)  # € / (lifetime * kWh)
        cost_buy_ele = self.Settings["cost_buy"]  # €/kWh

        cost_HeatStorage = self.Settings["cost_HeatStorage"] # €/kWh

        # input processing for time series
        time_step_selection = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
                               "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
                               "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
                               "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
        day = random.randint(120, 182)  # May - July
        day_type = 1

        # get an electric demand time series
        e_dem_obj = dtso.demand_time_series_object(shape=self.dem_shape,
                                                   absolute_demand=self.Settings["dem"],
                                                   delta_time=delta_time,
                                                   length=time.__len__(),
                                                   data_path="timeseries/e_dem.csv",
                                                   day_selection=day,
                                                   time_step_selection=time_step_selection,
                                                   day_type_selection="Day")

        DemandVal = e_dem_obj.get_array()

        # get a heat demand time series
        h_dem_obj = dtso.demand_time_series_object(shape=self.dem_shape,
                                                   absolute_demand=self.Settings["dem_heat"],
                                                   delta_time=delta_time,
                                                   length=time.__len__(),
                                                   data_path="timeseries/h_dem.csv",
                                                   day_selection=e_dem_obj.get_day_type() - 1,
                                                   time_step_selection=time_step_selection)

        DemandHeatVal = h_dem_obj.get_array()

        # get a PV availability time series with clouds
        pv_av_obj = ptso.pv_time_series_object(shape=self.pv_curve,
                                               delta_time=delta_time,
                                               length=time.__len__(),
                                               surplus=self.Settings["pow_pv_surplus"],
                                               demand_ts=e_dem_obj,
                                               data_path="timeseries/PV_TS.csv",
                                               day_selection=day,
                                               clouds=self.Settings["num_clouds"],
                                               cloud_size=self.Settings["pow_clouds"],
                                               cloud_dist=self.cloud_dist,
                                               cloud_shape=self.cloud_pow,
                                               pv_limit=[42, 126])

        availability_pv = pv_av_obj.get_array()

        #
        if isinstance(self.Settings["pow_clouds"], list):
            loss = self.Settings["pow_clouds"][0]
        else:
            loss = self.Settings["pow_clouds"]

        # Define the Pyomo variables

        # Electricity Sector
        model.EnergyPV = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyPV_curt = pyo.Var(time, within=pyo.NonNegativeReals)
        model.Demand = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyBattery = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyBattery_IN = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyBattery_OUT = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyBuy = pyo.Var(time, within=pyo.NonNegativeReals)
        model.CapacityBattery = pyo.Var(within=pyo.NonNegativeReals)
        # indicators for evaluation purpose only
        model.CostBuy = pyo.Var(within=pyo.Reals)
        model.CostBat = pyo.Var(within=pyo.Reals)
        model.NumClouds = pyo.Var(within=pyo.Reals)
        model.PV_TS = pyo.Var(time, within=pyo.Reals)
        model.PV_surplus = pyo.Var(within=pyo.Reals)
        model.energy_loss = pyo.Var(within=pyo.Reals)
        # Heat Sector
        model.DemandHeat = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyHeatStorage = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyHeatStorage_IN = pyo.Var(time, within=pyo.NonNegativeReals)
        model.EnergyHeatStorage_OUT = pyo.Var(time, within=pyo.NonNegativeReals)
        model.CapacityHeatStorage = pyo.Var(within=pyo.NonNegativeReals)
        model.HeatPump_Heat = pyo.Var(time, within=pyo.NonNegativeReals)
        model.HeatPump_Electricity = pyo.Var(time, within=pyo.NonNegativeReals)
        # indicators
        model.costHeatStorage = pyo.Var(within=pyo.Reals)

        # Define Objective
        model.cost = pyo.Objective(expr=cost_buy_ele * sum(
            model.EnergyBuy[i] for i in time) + cost_Battery * model.CapacityBattery +
                                        cost_HeatStorage * model.CapacityHeatStorage, sense=pyo.minimize)

        # Define Constraints
        model.limEQ = pyo.ConstraintList()  # technology limits

        for i in time:
            model.limEQ.add(
                model.EnergyPV[i] + model.EnergyPV_curt[i] == availability_pv[i])  # PV cap is assumed to be 1

        for i in time:
            model.limEQ.add(model.EnergyBattery[i] <= model.CapacityBattery)  # Battery UB

        for i in time:
            model.limEQ.add(model.EnergyHeatStorage[i] <= model.CapacityHeatStorage)  # HeatStorage energy bound

        model.limEQ.add(
            model.CapacityHeatStorage <= 1000 * 0.00419 * 0.277778 * 40)  # HeatStorage UB:liter * MJ/kg K * kWh/MJ * K

        for i in time:
            model.limEQ.add(model.HeatPump_Heat[i] <= 2 * max(DemandHeatVal))  # HeatPump UB

        model.InitialBattery = pyo.Constraint(
            expr=model.EnergyBattery[0] == model.EnergyBattery[time[-1]] - model.EnergyBattery_OUT[0] +
                 0.95 * model.EnergyBattery_IN[0])  # Battery level t=0 == t=T

        model.InitialHeatStorage = pyo.Constraint(
            expr=model.EnergyHeatStorage[0] == pow(0.99, delta_time) * model.EnergyHeatStorage[time[-1]] -
                 model.EnergyHeatStorage_OUT[0] +
                 model.EnergyHeatStorage_IN[0])  # Battery level t=0 == t=T

        model.DemandEQ = pyo.ConstraintList()

        for i in time:
            model.DemandEQ.add(expr=model.Demand[i] == DemandVal[i])  # Electricity Demand

        model.HeatDemandEQ = pyo.ConstraintList()

        for i in time:
            model.HeatDemandEQ.add(expr=model.DemandHeat[i] == DemandHeatVal[i])  # Heat Demand

        model.batteryEQ = pyo.ConstraintList()

        for i in time[1:]:
            model.batteryEQ.add(
                expr=model.EnergyBattery[i] == model.EnergyBattery[i - 1] - model.EnergyBattery_OUT[i] +
                     0.95 * model.EnergyBattery_IN[i])  # Battery Equation

        model.heatStorageEQ = pyo.ConstraintList()

        for i in time[1:]:
            model.heatStorageEQ.add(
                expr=model.EnergyHeatStorage[i] == pow(0.99, delta_time) * model.EnergyHeatStorage[i - 1] -
                     model.EnergyHeatStorage_OUT[i] +
                     model.EnergyHeatStorage_IN[i])  # Heat Storage Equation

        model.EnergyEQ = pyo.ConstraintList()

        for i in time:
            model.EnergyEQ.add(
                expr=model.Demand[i] == model.EnergyBuy[i] + model.EnergyBattery_OUT[i] - model.EnergyBattery_IN[
                    i] + model.EnergyPV[i] - model.HeatPump_Electricity[i])  # Energy Equation

        model.HeatPumpEQ = pyo.ConstraintList()

        for i in time:
            model.HeatPumpEQ.add(
                expr=model.HeatPump_Heat[i] <= model.HeatPump_Electricity[i] * self.Settings["HP_COP"]
            )  # Heat Pump Equation

        model.HeatEQ = pyo.ConstraintList()

        for i in time:
            model.HeatEQ.add(
                expr=model.DemandHeat[i] == model.EnergyHeatStorage_OUT[i] - model.EnergyHeatStorage_IN[i] +
                     model.HeatPump_Heat[i]
            )

        # Some equations that store input settings in a Variable
        model.ValueCostBuy = pyo.Constraint(expr=model.CostBuy == cost_buy_ele)
        model.ValueCostHS = pyo.Constraint(expr=model.costHeatStorage == cost_HeatStorage)
        model.ValueCostBat = pyo.Constraint(expr=model.CostBat == cost_Battery)
        model.ValueSurplus = pyo.Constraint(expr=model.PV_surplus == self.Settings["pow_pv_surplus"])
        model.ValueLoss = pyo.Constraint(expr=model.energy_loss == loss)
        model.NumberClouds = pyo.Constraint(expr=model.NumClouds == self.Settings["num_clouds"])
        model.PV = pyo.ConstraintList()
        for i in time:
            model.PV.add(model.PV_TS[i] == availability_pv[i])

        self.model = model

    def run(self):
        """Run a build model. Changes might be necessary to run with a different solver"""
        model = self.model
        solver = SolverFactory('cplex')
        # some solver settings
        solver.options["emphasis_numerical"] = 'y'
        # solver.options["lpmethod"] = 4
        solver.options["simplex_tolerances_optimality"] = 1e-6
        results = solver.solve(model, tee=True, keepfiles=True)
        results.write()
        # model.pprint()
        return model, results.solver.termination_condition

    def build_and_run(self):
        self.build()
        return self.run()


def getKPI(self, basemodel=None):
    """read the values from a solved model and return them as a dictionary"""

    d_x_i = {"c_b": pyo.value(self.CostBat),
             "pv": [pyo.value(self.PV_TS[i]) for i in range(144)],
             "c_HS": pyo.value(self.costHeatStorage)}

    ResDict = {}
    ResDict["Battery"] = pyo.value(self.CapacityBattery)
    ResDict["HeatStorage"] = pyo.value(self.CapacityHeatStorage)
    ResDict["c_Bat"] = pyo.value(self.CostBat)
    ResDict["surplus"] = pyo.value(self.PV_surplus)
    ResDict["clouds"] = pyo.value(self.NumClouds)
    ResDict["energy_loss"] = pyo.value(self.energy_loss)
    ResDict["c_HS"] = pyo.value(self.costHeatStorage)
    ResDict["d_b"] = d_x_i["c_b"]
    ResDict["d_p"] = d_x_i["pv"]
    ResDict["d_h"] = d_x_i["c_HS"]
    if basemodel is not None:
        bm_kpi = getKPI(basemodel)
        ResDict["d_b"] = d_x_i["c_b"] - bm_kpi["d_b"]
        ResDict["d_p"] = np.linalg.norm(np.array(d_x_i["pv"]) - np.array(bm_kpi["d_p"]))
        ResDict["d_h"] = d_x_i["c_HS"] - bm_kpi["d_h"]
    return ResDict
