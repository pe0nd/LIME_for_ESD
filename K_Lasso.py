"""performe a distance based lasso that returns k parameters"""


def k_lasso(interpretable_data, distance, target, k, kernel="RBF", sample_idx=-1,  limit=1e-3):
    # calculate the weighted lasso
    if sample_idx!=-1:
        if kernel=="RBF":
            def calc_dist(sample_idx, data):
                import numpy as np
                from sklearn.gaussian_process.kernels import RBF
                import pandas as pd
                # some distance function (here: exponential distance)
                kernel = RBF(length_scale=np.array(data.std()), length_scale_bounds=(0.0, 1.0))
                #sample_dist = (np.sqrt(np.square(data - data.iloc[sample_idx])).sum(axis=1))
                #sample_dist = np.exp(-sample_dist / (2* np.square(np.std(sample_dist))))
                sample_dist = kernel(data)[sample_idx]
                return sample_dist
            dist = calc_dist(sample_idx, distance)
        else:
            dist = kernel(distance)[sample_idx]
    else:
        dist=distance

    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    import numpy as np
    from sklearn import linear_model



    relevant_results = k+1
    ld = 1
    r = []
    while relevant_results > k and ld < 200:
        a = 0.1 * np.exp(0.1*ld) - 0.1
        clf = linear_model.Ridge(alpha=a)
        clf.fit(interpretable_data, target, sample_weight=dist)
        r= clf.coef_
        # model = pyo.ConcreteModel()
        # model.weights = pyo.Var(range(interpretable_data.columns.__len__()), within=pyo.Reals)
        # model.abs = pyo.Var(range(interpretable_data.columns.__len__()), within=pyo.Reals)
        # model.obj = pyo.Objective(
        #     expr=sum(np.square(dist[i] * (target[i] -
        #                                        sum(model.weights[j] * interpretable_data.iloc[i][j] for j in
        #                                            range(interpretable_data.columns.__len__()))
        #                                        )) for i in range(target.__len__())) + ld * sum(
        #         model.abs[l] for l in range(interpretable_data.columns.__len__())),
        #     sense=pyo.minimize)
        # model.AbsEq = pyo.ConstraintList()
        #
        # for i in range(interpretable_data.columns.__len__()):
        #     model.AbsEq.add(expr=model.abs[i] >= model.weights[i])
        #     model.AbsEq.add(expr=model.abs[i] >= -model.weights[i])
        #
        # solver = SolverFactory('cplex')
        # solver.options["emphasis_numerical"] = 'y'
        # # solver.options["lpmethod"] = 4
        # solver.options["simplex_tolerances_optimality"] = 1e-6
        #results = solver.solve(model, tee=False)
        #results.write()
        # model.pprint()
        #from pyomo.opt import SolverStatus, TerminationCondition
        #r = [pyo.value(model.weights[i]) for i in range(interpretable_data.columns.__len__())]
        relevant_results = sum(int((abs(r[i])-limit) >= 0) for i in range(interpretable_data.columns.__len__()))
        ld += 1
    #print(ld)
    return r
