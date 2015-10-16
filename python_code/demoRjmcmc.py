import numpy as np
# import matplotlib.pyplot as plt
from rjmcmc import *
from normalise import *
from samples import *


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

lambda1, lambda2, states = posterior_finiteMixture(y, K, extremes, 10000)

state1 = sum(states == 1)
state2 = sum(states == 2)

############################
# likelihood
#############################
iter = int(1e6)
lambda12 = jeffreysPrior(iter, extremes)
lambda22 = jeffreysPrior(iter, extremes)
pi = np.random.beta(alpha, alpha, (iter, 1))
log_py = np.zeros((iter, 1))
normC = np.diff(np.log(extremes))

l = 0
for i in range(iter):
    a = np.log(pi[i])+np.log(lambda12[i])-lambda12[i]*y
    b = np.log(1-pi[i])+np.log(lambda22[i])-lambda22[i]*y
    log_pyi = np.column_stack((a, b))
    log_py[i] = sum(logsumexp(log_pyi, 1))

# a = np.log(pi)+np.log(lambda12)-lambda12*y.T
# b = np.log(1-pi)+np.log(lambda22)-lambda22*y.T

log_py_k2 = logsumexp(log_py, 0)-np.log(iter)
log_py_k1 = -np.log(normC)-N*np.log(sum(y))+sp.gammaln(N)+np.log(np.diff(sp.gammainc(N, sum(y)*extremes)))

p_k1 = 1./(1.+np.exp(log_py_k2-log_py_k1))
print(p_k1)
print(state1/float(state1+state2))
# plt.figure()
# plt.plot(lambda1)
# plt.show()
# plt.figure()
# plt.plot(lambda2[:, 0], lambda2[:, 1], '.')
# plt.show()

