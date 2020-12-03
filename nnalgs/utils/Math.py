from math import sqrt

import numpy as np


# tools used in production
# ---------------------------------------------------------------------------------------------
def combine_mean_std(l_n, l_mean, l_std):
    """
    Ref: https://www.statstodo.com/index.php and
     Altman DG, Machin D, Bryant TN and Gardner MJ. (2000)
     Statistics with Confidence Second Edition.
     BMJ Books ISBN 0 7279 1375 1. p. 28-31
    :param l_n: the sample size of each group
    :param l_mean: the mean value of each group
    :param l_std: the standard deviation value of each group
    :return: Combined sample size, mean value and standard deviation value
    """
    l_sum_x = [n * mean for n, mean in zip(l_n, l_mean)]
    l_sum_x2 = [sd ** 2 * (n - 1) + sum_x ** 2 / n for n, sd, sum_x in zip(l_n, l_std, l_sum_x)]
    tn, tx, txx = sum(l_n), sum(l_sum_x), sum(l_sum_x2)

    return tn, tx / tn, sqrt((txx - tx ** 2 / tn) / (tn - 1))


def remove_eta_phi(x):
    """
    used by algs `DecayModeLSTMNoEtaPhi`
    :param x: torch.Tensor
    :return:
    """
    return x[:, :, [i for i in range(x.size()[2]) if i != 0 and i != 2]].contiguous()


# useful NumPy manipulation, not used in production (!)
# ---------------------------------------------------------------------------------------------
def log10(arr, epsilon=0.0):
    """

    :param arr: np.ndarray
    :param epsilon: constant preventing log10(0)
    :return: np.ndarray
    """
    masked = np.ma.masked_equal(arr, 0)
    masked = np.log10(np.maximum(masked, epsilon))
    return masked.filled(0)
