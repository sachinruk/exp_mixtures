import numpy as np
from rjmcmc import *
from normalise import *

alpha = 5
K = 2
N = 50
a = 1
b = 1


np.random.seed(1)
pi = np.array([0.4, 0.6])
lambda_ = np.array([2, 6])
z = np.random.multinomial(1, pi, N)
y = np.random.gamma(1, 1.0/np.dot(z, lambda_)).reshape(N, 1)
extremes = [(1/y).min(), (1/y).max()]

lambda_chain = posterior_finiteMixture(y, K, extremes, 200)
