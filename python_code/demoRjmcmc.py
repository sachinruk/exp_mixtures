import numpy as np
from rjmcmc import *
from normalise import *
import matplotlib.pyplot as plt

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

lambda_chain = posterior_finiteMixture(y, K, extremes, 2000)

print(lambda_chain)

#  separate the lambda1s and 2s
lambda1 = [val for val in lambda_chain if len(val) == 1]
lambda2 = np.array([val for val in lambda_chain if len(val) == 2])

plt.figure()
plt.plot(lambda1)
plt.show()
plt.figure()
plt.plot(lambda2[:, 0], lambda2[:, 1], '.')
plt.show()

