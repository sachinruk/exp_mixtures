import numpy as np


def logsumexp(x, axis):
    mx = np.max(x, axis)
    xshift = [x[i]-mx[i] for i in range(len(x))]
    return np.log(np.sum(np.exp(xshift), axis)) + mx
