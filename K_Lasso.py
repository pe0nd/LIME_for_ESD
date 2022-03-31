import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model


def calc_dist(sample_idx, data):  # RBF kernel
    kernel = RBF(length_scale=np.array(data.std()), length_scale_bounds=(0.0, 1.0))
    sample_dist = kernel(data)[sample_idx]
    return sample_dist

def k_lasso(interpretable_data, distance, target, k, kernel="RBF", sample_idx=-1, limit=1e-3):
    """
    Perform a distance based lasso that returns weights for all parameters,
    with the k most relevant having an absolute weight above limit

    Args:
        interpretable_data(list): list of samples with different feature variations
        distance(list): distance between samples
        target(list): parameter that should be explained/ target of regression
        k(int): number of feature weights that should be above "limit", hyperparameter will be adjusted accordingly
        kernel(function or string): a kernel function that is used with the values of "distance"
        sample_idx(int): index of base sample. If -1 the kernel will not be applied,
                        but distance will be used as wights directly
        limit(float): limit for which k will be seen as irrelevant for explanation

    """
    # get weights
    if sample_idx != -1:
        if kernel == "RBF":
            dist = calc_dist(sample_idx, distance)
        else:  # External kernel. Expected to work like the kernels from sklearn.
            dist = kernel(distance)[sample_idx]
    else:  # External weights
        dist = distance

    # do the ridge regression which changing hyperparameter until only k parameters have an absolute weight
    # higher than limit
    relevant_results = k + 1
    ld = 1
    r = []
    while relevant_results > k and ld < 200:
        a = 0.1 * np.exp(0.1 * ld) - 0.1  # update hyperparameter
        clf = linear_model.Ridge(alpha=a)
        clf.fit(interpretable_data, target, sample_weight=dist)
        r = clf.coef_
        relevant_results = sum(int((abs(r[i]) - limit) >= 0) for i in range(interpretable_data.columns.__len__()))
        ld += 1
    return r
