import numpy as np


def jeffreysPrior(N, extremes):
    normC = -np.diff(np.log(extremes))
    u = np.random.rand(N,1)
    x = np.exp(u*normC+np.log(extremes[0]))
    return x
