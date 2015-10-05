import os
os.chdir('/home/sachin/Documents/thesis/exp_mixtures/python_code/')

import numpy as np
from dirichlet_sample import *
from posterior_finiteMixture import *
from posterior_restricted import *
from normalise import *

import time

import matplotlib.pyplot as plt

alpha=5;
# K=10; 
K=2;
N=50;
a=1; b=1;


np.random.seed(1)
pi=np.array([0.4, 0.6]);
# lambda_=gamrnd(a,1/b,K,1);
lambda_=np.array([2, 6]);
z=np.random.multinomial(1,pi,N);
y=np.random.gamma(1,1.0/np.dot(z,lambda_)).reshape(N,1);

#t = time.time()
#elapsed = time.time() - t
#print elapsed

#run with unrestricted prior
lambda_, pi=posterior_finiteMixture(y,K,10000);
plt.figure()
plt.plot(lambda_[:,0],lambda_[:,1],'.')
plt.figure()
plt.hist2d(lambda_[:,0],lambda_[:,1],bins=50)
plt.title('Unrestricted prior')

#run with restricted prior
extremes=[(1/y).min(), (1/y).max()]
lambda_, pi=posterior_restricted(y,K,extremes,10000);
plt.figure()
plt.plot(lambda_[:,0],lambda_[:,1],'.')
plt.figure()
plt.hist2d(lambda_[:,0],lambda_[:,1],bins=50)
plt.title('Restricted prior')

#run with restricted - cheat method
extremes=[0.5, 15]
lambda_, pi=posterior_restricted(y,K,extremes,10000);
plt.figure()
plt.plot(lambda_[:,0],lambda_[:,1],'.')
plt.figure()
plt.hist2d(lambda_[:,0],lambda_[:,1],bins=50)
plt.title('Restricted prior - Cheat method')

plt.show()

