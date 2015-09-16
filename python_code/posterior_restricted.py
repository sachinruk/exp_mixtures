from scipy import special as sp
import numpy as np
from normalise import *
from multi_sample import *
def posterior_finiteMixture(y,K,iter):

    # N=length(y);
    a=0.1; b=0.1; alpha=5;
    lambda_=np.zeros((iter+1,K));
    pi=np.zeros((iter+1,K));
    lambda_[0]=np.random.gamma(1,1,K);
    pi[0]=np.random.dirichlet(alpha*np.ones((1,K))[0],1)
    
    N=len(y)
    
    for i in range(1,len(pi)):
        #z variable
        z=q_z(y,pi[i-1],lambda_[i-1])
        n_k=sum(z);
        #lambda_ variable
        lambda_[i]=q_lambda(y,z,N,n_k,a,b)
        #pi variable
        pi[i]=q_pi(n_k,alpha)
    return lambda_,pi

# The posteriors    
def q_z(y,pi,lambda_):
    p_z=-y*lambda_+np.log(pi*lambda_);
    p_z=normalise(p_z);
    z=categorical_sample(p_z);
    return z
    
def q_lambda(y,z,N,n_k,a,b):
    gam_a=a+n_k; gam_b=b+sum(z*y);
    u=np.random.uniform(0,1,(N,1));
    F_max=sp.gammainc(gam_a,gam_b*y.max()); 
    F_min=sp.gammainc(gam_a,gam_b*y.min());
    lambda_const=F_max-F_min;
    return sp.gammaincinv(gam_a,F_min+u*lambda_const);    
    
def q_pi(n_k,alpha):
    dir_par=alpha+n_k;
    return np.random.dirichlet(dir_par,1);    
