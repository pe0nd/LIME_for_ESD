import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn import linear_model
import warnings as wr


def calc_dist(sample_idx, data):  # RBF kernel
    if 0 in np.array(data.std()):
        wr.warn("The standard deviation of an data series is 0. This series will be ignored for distance calculation.")
    data = np.array(data)[:, np.array(data.std()) != 0]
    kernel = RBF(length_scale=np.array(data.std(axis=0)), length_scale_bounds=(0.0, 1.0))
    sample_dist = kernel(data)[sample_idx]
    return sample_dist


def k_lasso(interpretable_data, distance, target, k, kernel="RBF", sample_idx=-1, limit=1e-3, method='lasso_path'):
    """
    Perform a distance based lasso that returns weights for all parameters,
    with the k most relevant having an absolute weight above limit

        :param limit: limit for which k will be seen as irrelevant for explanation
        :param sample_idx: index of base sample. If -1 the kernel will not be applied,
                        but distance will be used as wights directly
        :param kernel: a kernel function that is used with the values of "distance"
        :param k: number of feature weights that should be above "limit", hyperparameter will be adjusted accordingly
        :param target: parameter that should be explained/ target of regression
        :param distance: distance between samples
        :param interpretable_data: list of samples with different feature variations
        :param method: method for the regression, either "lasso_path" or "ridge"

    """
    # get weights
    if sample_idx != -1:
        if kernel == "RBF":
            # the sample is needed since the kernel calculates the distance between all points
            # and we are only interested to the distance to the main sample
            dist = calc_dist(sample_idx, distance)
        else:
            # External kernel. Expected to work like the kernels from sklearn.
            dist = kernel(distance)[sample_idx]
    else:
        # External weights
        dist = distance

    if method == 'lasso_path':
        # regression using the lasso method (absolute penalty term)
        weighted_data = ((interpretable_data - np.average(interpretable_data, axis=0,
                                                          weights=dist))
                         * np.sqrt(dist[:, np.newaxis]))
        weighted_labels = ((target - np.average(target, axis=0, weights=dist))
                           * np.sqrt(dist[:]))

        alphas, _, coefs = linear_model.lars_path(weighted_data.to_numpy(),
                                                  weighted_labels.to_numpy(),
                                                  method='lasso',
                                                  verbose=False)

        for i in range(coefs.shape[1]):
            if sum(abs(coefs.T)[i] > 0) == k:
                return coefs.T[i]

    if method == 'ridge':
        # regression using the ridge regression (quadratic penalty term)
        relevant_results = k + 1
        ld = 1
        r = []
        while relevant_results > k and ld < 200:
            # update hyperparameter
            a = 0.1 * np.exp(0.1 * ld) - 0.1
            clf = linear_model.Ridge(alpha=a)
            clf.fit(interpretable_data, target, sample_weight=dist)
            r = clf.coef_
            relevant_results = sum(int((abs(r[i]) - limit) >= 0) for i in range(interpretable_data.columns.__len__()))
            ld += 1
        return r
