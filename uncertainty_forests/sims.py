"""
sims.py
-------
This file contains simulation code.
"""

import numpy as np


def get_sample(n, mean, var):
    """
    This is the most basic simulated samples. The following
    data is generated:
    X ~ Normal(y*mean, var)
    y is 1 or -1 with probability 1/2
    """

    x_sample = []
    y_sample = []
    for i in range(n):
        y = np.random.binomial(1, .5)
        if (y == 0):
            x = np.random.normal(-mean, var)
        else:
            x = np.random.normal(mean, var)
        x_sample.append(x)
        y_sample.append(y)
    # the final reshape is for sklearn to be able to train correctly
    return np.array(x_sample).reshape(-1, 1), y_sample


def get_multivariate_sample(n, d, mean):
    """
    Multivariate extension for sampling code above.
    Each additional dimension (specified by d) is independent Normal(0, 1). 
    """
    x_sample = []
    y_sample = []
    means = np.zeros(d)
    means[0] = mean
    for i in range(n):
        y = np.random.binomial(1, .5)
        if (y == 0):
            x = np.random.multivariate_normal(-means, np.identity(d))
        else:
            x = np.random.multivariate_normal(means, np.identity(d))
        x_sample.append(x.tolist())
        y_sample.append(y)
    
    return np.array(x_sample), y_sample