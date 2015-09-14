import os
os.chdir('/home/sachin/thesis/exp_mixtures/python_code/')

import numpy as np
from dirichlet_sample import *
from posterior_finiteMixture import *
from normalise import *

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
y=np.random.gamma(1,1.0/np.dot(z,lambda_));

lambda_, pi=posterior_finiteMixture(y,K,100);