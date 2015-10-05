import numpy as np
from normalise import *
from multi_sample import *
def posterior_finiteMixture(y,K,iterations):

    # N=length(y);
    a=0.1; b=0.1; alpha=5;
    lambda_=np.zeros((iterations+1,K));
    pi=np.zeros((iterations+1,K));
    lambda_[0]=np.random.gamma(1,1,K);
    pi[0]=np.random.dirichlet(alpha*np.ones((1,K))[0],1)


    for i in range(1,iterations):
        if state==1:
            lambda1=lambda_[i]
            mu1,mu2=np.random.uniform(0,1,2)
            lambda2[0]=lambda1*mu1/(1-mu1)
            lambda2[1]=lambda1*(1-mu1)/mu1
            pi_12=[mu2,1-mu2]
            z=q_z(y,pi_12,lambda2)
            chosen_lambda=z*lambda2
            log_joint_lik2=(np.sum(np.log(chosen_lambda)-chosen_lambda*y)
                            -2*np.log(normC)-np.log(lambda2[0])-np.log(lambda2[0])
                            -np.log(K))
         else:
             lambda2=lambda_[i]
             lambda1=lambda2.prod()
             log_joint_lik1=(N*np.log(lambda1)-np.sum(lambda1*y)
                            -np.log(normC)-np.log(lambda1)
                            -np.log(K))
                            

	q_theta2_recip=2*lambda1/(mu1*(1-mu1))
        alpha_ratio=joint_ratio(y,lambda2,lambda1,pi_12)*q_theta2_recip
	if state==2:

	A=min(1,
	if A>np.random.uniform():
            #accept move
        else:
            #keep old value
    
    return lambda_,pi


    
def q_z(y,pi,lambda_):
    p_z=-y*lambda_+np.log(pi*lambda_);
    p_z=normalise(p_z);
    z=categorical_sample(p_z);
    return z
