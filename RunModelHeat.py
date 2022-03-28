import pandas


def run():
    import ModelHeat
    import pyomo.environ as pyo
    import csv
    import numpy as np
    # get settings for HouseModel

    pv_c = ["real"]  # , "3box", "par"
    cloud_distrib = ["equal"]  # ,"random_hour"
    cloud_shape = ["dist"]  # "pow",dist

    settings = ModelHeat.getSettings()
    settings["cost_Battery"] = 60 / 365
    settings["num_clouds"] = 5
    settings["pow_pv_surplus"] = 15
    settings["pow_clouds"] = 5 * 0.1
    settings["cost_HeatStorage"] = 5 / 365
    if cloud_shape[0] == "dist":
        settings["pow_clouds"] = [5 * 0.1, 5 * 0.01]

    HousePVModel = ModelHeat.HouseModel(pv_curve="real", cloud_dist="equal",
                                        cloud_pow="dist", non_reductive=True,
                                        dem_shape="real", dem_heat_shape="real",
                                        settings_dict=settings)

    [baseResult, Status] = HousePVModel.sample_model()


    for p_c in pv_c:
        for c_d in cloud_distrib:
            for c_s in cloud_shape:
                # print(str(p_c) + '_' + str(c_d) + '_' + str(c_s))
                ResAll = {}
                iteration_count = -1

                for c_bat in [54, 60, 66]:  #[114, 120, 126]:
                    for surplus in [14.5, 15, 15.5]:
                        for clouds in range(4, 7, 1):
                            for loss in range(4, 7, 1):
                                for c_HS in [4, 5, 6]:#[172, 180, 188]:  # random permutations
                                    for rnd in range(5):
                                        iteration_count += 1
                                        #print(str(p_c) + '_' + str(c_d) + '_' + str(c_s))
                                        #print(str(iteration_count) + " of " + str(4 * 6 * 11 * 5 * 4))

                                        if not (c_d == "equal" and c_s == "pow" and rnd > 0):

                                            settings = ModelHeat.getSettings()
                                            settings["cost_Battery"] = c_bat / 365
                                            settings["num_clouds"] = clouds
                                            settings["pow_pv_surplus"] = surplus
                                            settings["pow_clouds"] = loss * 0.1
                                            settings["cost_HeatStorage"] = c_HS / 365
                                            if c_s == "dist":
                                                settings["pow_clouds"] = [loss * 0.1, loss * 0.01]

                                            # Create an instance of the HouseModel with given (or modified) settings
                                            HousePVModel = ModelHeat.HouseModel(pv_curve=p_c, cloud_dist=c_d,
                                                                                cloud_pow=c_s, non_reductive=True,
                                                                                dem_shape="real",
                                                                                dem_heat_shape="real",
                                                                                settings_dict=settings)

                                            [Result, Status] = HousePVModel.sample_model()

                                            ResAll[ResAll.__len__()] = ModelHeat.getKPI(Result, basemodel=baseResult)

                with open('LocalSamples_Heat_lp_vlh_' + str(p_c) + '_' + str(c_d) + '_' + str(c_s) + '.csv', 'w',
                          newline='') as csvfile:
                    fieldnames = ['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss', 'Battery', 'HeatStorage',
                                  "d_b", "d_p", "d_h"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

                    writer.writeheader()
                    for i in ResAll:
                        writer.writerow(ResAll[i])


if __name__ == '__main__':
    run()

    # import csv
    #
    # ResAll = {}
    # with open('LocalSamples_Heat_box_equal_dist.csv', newline='') as csvfile:
    #     reader = csv.DictReader(csvfile, delimiter=';')
    #     for row in reader:
    #         ResAll[ResAll.__len__()] = {"c_Bat": float(row["c_Bat"]), "c_HS": float(row["c_HS"]),
    #                                     "surplus": float(row["surplus"]),
    #                                     "clouds": float(row["clouds"]), "energy_loss": float(row["energy_loss"]),
    #                                     "Battery": float(row["Battery"]), "HeatStorage": float(row["HeatStorage"])}
    #
    # import plotly.express as px
    # import pandas as pd
    # from sklearn import linear_model
    #
    # Res = pd.DataFrame(ResAll).transpose()
    # Res = Res.groupby(["c_Bat", "c_HS", "surplus", "clouds", "energy_loss"], as_index=False).agg("mean")
    #
    # # Res_c_low = Res.loc[Res["c_Bat"] <= 60]
    # # Res_c_high = Res.loc[Res["c_Bat"] > 60]
    # #
    # # show_data = Res_c_high
    #
    # # fig = px.scatter(show_data, x="surplus", y="Battery", color="c_Bat",
    # #                  size='energy_loss', hover_data=['clouds'])
    # # fig.show()
    # #
    # # fig = px.scatter(show_data, x="energy_loss", y="Battery", color="c_Bat",
    # #                  size='surplus', hover_data=['clouds'])
    # # fig.show()
    # #
    # # fig = px.scatter(show_data, x="clouds", y="Battery", color="c_Bat",
    # #                  size='surplus', hover_data=['energy_loss'])
    # # fig.show()
    # from sklearn import preprocessing
    #
    # normalized_Res = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(Res), columns=Res.columns)
    #
    # # combined model
    # # clf = linear_model.Lasso(alpha=0.1)
    # # clf.fit(normalized_Res[["c_Bat", "c_HS", "surplus", "clouds", "energy_loss", "HeatStorage"]], normalized_Res["Battery"])
    # # print("Lasso of all Data")
    # # print(clf.coef_)
    # #
    # # # split by c_Bat
    # #
    # # # normalize again
    # # normalized_Res_c_low = (Res_c_low - Res_c_low.mean()) / Res_c_low.std()
    # # normalized_Res_c_high = (Res_c_high - Res_c_high.mean()) / Res_c_high.std()
    # #
    # # clf_low = linear_model.Lasso(alpha=0.1)
    # # clf_low.fit(normalized_Res_c_low[["c_HS","surplus", "clouds", "energy_loss", "HeatStorage"]], Res_c_low["Battery"])
    # # print("Lasso of low price")
    # # print(clf_low.coef_)
    # #
    # # clf_high = linear_model.Lasso(alpha=0.1)
    # # clf_high.fit(normalized_Res_c_high[["c_HS","surplus", "clouds", "energy_loss", "HeatStorage"]], Res_c_high["Battery"])
    # # print("Lasso of high price")
    # # print(clf_high.coef_)
    #
    # # weighting the entrys and explaining it locally
    # import random
    #
    # data = normalized_Res
    # data_names = Res
    #
    # instance_idx = random.randint(0, data.__len__() - 1)
    # instance_idx = 122
    #
    # import K_Lasso as Kl
    #
    # r = Kl.k_lasso(instance_idx, data[['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss']],
    #                data['Battery'], 1)
    # rh = Kl.k_lasso(instance_idx, data[['c_Bat', 'c_HS', 'surplus', 'clouds', 'energy_loss']],
    #                 data['HeatStorage'], 1)
    #
    # print("Explanation for Sample " + str(instance_idx) +
    #       # " Price: " + str(data_names.iloc[instance_idx].loc["c_Bat"]) +
    #       " with Surplus: " + str(data_names.iloc[instance_idx].loc["surplus"]) + " clouds: " + str(
    #     data_names.iloc[instance_idx].loc["clouds"]) + " loss per cloud: " + str(
    #     data_names.iloc[instance_idx].loc["energy_loss"]))
    # print([round(r[i], 4) for i in range(r.__len__())])
