from scipy import special as sp
import numpy as np
from normalise import *
from multi_sample import *
# The posteriors


def q_z2(y, pi, lambda_):
    p_z = -y*lambda_+np.log(pi*lambda_)
    p_z = normalise(p_z)
    z = categorical_sample(p_z)
    return z


def q_lambda2(y, z, n_k, extremes):
    gam_a = n_k; gam_b = sum(z*y)
    u = np.random.uniform(0, 1, len(n_k))
    F_min, F_max = sp.gammainc(gam_a, gam_b*np.array(extremes))
    lambda_const = F_max-F_min

    lambda2 = sp.gammaincinv(gam_a, F_min+u*lambda_const)/gam_b

    idx = np.where(gam_a == 0)
    if len(idx[0]):  # if any values with gam_a==0
        F_min, F_max = np.log(extremes)
        normC = F_max-F_min
        lambda2[idx] = np.exp(u[idx]*normC+F_min)
    return lambda2


def q_pi2(n_k, alpha):
    dir_par = alpha+n_k
    return np.random.dirichlet(dir_par, 1)


def q_lambda(y, extremes):
    gam_a = len(y); gam_b = sum(y)
    u = np.random.uniform()
    F_min, F_max = sp.gammainc(gam_a, gam_b*np.array(extremes))
    lambda_const = F_max-F_min
    return sp.gammaincinv(gam_a, F_min+u*lambda_const)/gam_b
