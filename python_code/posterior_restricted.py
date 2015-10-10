import numpy as np
from posteriors import *
def posterior_restricted(y,K,extremes,iter):

    # N=length(y);
    a=0.1; b=0.1; alpha=5;
    lambda_=np.zeros((iter+1,K));
    pi=np.zeros((iter+1,K));
    lambda_[0]=np.random.gamma(1,1,K);
    pi[0]=np.random.dirichlet(alpha*np.ones((1,K))[0],1)
    
    
    
    for i in range(1,len(pi)):
        #z variable
        z=q_z2(y,pi[i-1],lambda_[i-1])
        n_k=sum(z);
        #lambda_ variable
        
        lambda_[i]=q_lambda2(y,z,n_k,a,b,extremes)
        #pi variable
        pi[i]=q_pi2(n_k,alpha)
    return lambda_,pi
