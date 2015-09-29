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
        mu1,mu2=np.random.uniform(0,1,2)
        lambda2_s[0]=lambda1*mu1/(1-mu1)
        lambda2_s[1]=lambda1*(1-mu1)/mu1
        pi_12=mu2
        
        A=min(1,)
        
        
        #z variable
        p_z=-y*lambda_[i-1]+np.log(pi[i-1]*lambda_[i-1]);
        p_z=normalise(p_z);
        z=categorical_sample(p_z);
        #lambda_ variable
        n_k=sum(z);
        gam_a=a+n_k; gam_b=b+sum(z*y);
        lambda_[i]=np.random.gamma(gam_a,1./gam_b);
        #pi variable
        dir_par=alpha+n_k;
        pi[i]=np.random.dirichlet(dir_par,1);
        
    return lambda_,pi
