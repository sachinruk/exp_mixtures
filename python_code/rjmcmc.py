import numpy as np
from normalise import *
from multi_sample import *
def posterior_finiteMixture(y,K,iterationsations):

    # N=length(y);
    a=0.1; b=0.1; alpha=5;
    lambda_=np.zeros((iterations+1,K));
    pi=np.zeros((iterations+1,K));
    lambda_[0]=np.random.gamma(1,1,K);
    pi[0]=np.random.dirichlet(alpha*np.ones((1,K))[0],1)


    for i in range(1,iterations):
        if state==1:
            lambda1_s=lambda_[i]
            mu1,mu2=np.random.uniform(0,1,2)
            lambda2_s[0]=lambda1_s*mu1/(1-mu1)
            lambda2_s[1]=lambda1_s*(1-mu1)/mu1
            pi_12=mu2
         else:
             lambda2_s=lambda_[i]
             lambda1_s=lambda2_s.prod()

	q_theta2_recip=2*lambda1/(mu1*(1-mu1))
        A=min(1,joint_ratio2(y,lambda2_s,lambda1_s,pi_12)*q_theta2_recip)
	if A>np.random.uniform():
            #accept move
        else:
            #keep old value
    
    return lambda_,pi
