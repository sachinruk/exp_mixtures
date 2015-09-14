import numpy as np
from normalise import *
def posterior_finiteMixture(y,K,iter):

    # N=length(y);
    a=0.1; b=0.1; alpha=5;
    lambda_=np.zeros((iter+1,K));
    pi=np.zeros((iter+1,K));
    lambda_[0]=np.random.gamma(1,1,K);
    pi[0]=np.random.dirichlet(alpha*np.ones((1,K))[0],1)


    for i in range(1,len(pi)):
        #z variable
        p_z=-np.dot(y,lambda_[i-1])+log(pi[i-1]*lambda_[i-1]);
        p_z=normalise(p_z);
        z=np.random.multinomial(1,p_z);
        #lambda_ variable
        n_k=sum(z);
        gam_a=a+n_k; gam_b=b+sum(z*y);
        lambda_[i]=np.random.gamma(gam_a,1./gam_b);
        #pi variable
        dir_par=alpha+n_k;
        pi[i]=np.random.dirichlet(dir_par,1);