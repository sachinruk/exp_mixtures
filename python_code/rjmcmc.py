import numpy as np
from normalise import *
from multi_sample import *
from posteriors import *
def posterior_finiteMixture(y,K,extremes,iterations):
    a=0.1; b=0.1; alpha=5;
    lambda_=np.zeros((iterations+1,K));
    pi=np.zeros((iterations+1,K));
    lambda_[0]=np.random.gamma(1,1,K);
    pi[0]=np.random.dirichlet(alpha*np.ones((1,K))[0],1)

    state=1

    lambdaChain=list()
    for i in range(1,iterations):
        if state==1:
            lambda1=lambda_[i]
            mu1,mu2=np.random.uniform(0,1,2)
            lambda2[0]=lambda1*mu1/(1-mu1)
            lambda2[1]=lambda1*(1-mu1)/mu1
            pi_12=[mu2,1-mu2]
            lik=np.sum(lambda2*pi_12*np.exp(y*lambda2),1)
            log_joint_lik2=(np.sum(np.log(lik))
                            -2*np.log(normC)-np.log(lambda2[0])-np.log(lambda2[0])
                            -np.log(K))
         else:
             lambda2=lambda_[i]
             lambda1=lambda2.prod()
             log_joint_lik1=(N*np.log(lambda1)-np.sum(lambda1*y)
                            -np.log(normC)-np.log(lambda1)
                            -np.log(K))

        #q_theta2_recip=2*lambda1/(mu1*(1-mu1))
        logq=np.log(2)+np.log(lambda1)-np.log(mu1)-np.log(1-mu1)
        alpha_ratio=log_joint_lik2-log_joint_lik1+logq
        A=min(1,np.exp(alpha_ratio))
        if state==2:
            A=min(1,np.exp(-alpha_ratio))

        if A>np.random.uniform(): #accept move
            if state==1:
                lambda1=q_lambda(y,extremes)
                lambdaChain.append(lambda1)
            else: #state 2
                z=q_z2(y,pi[i-1],lambda_[i-1])
                n_k=sum(z);
                lambda2=q_lambda2(y,z,n_k,a,b,extremes)
                pi[i]=q_pi2(n_k,alpha)

                lambdaChain.append(lambda2)
        else: #keep old value
            if state==1:
                lambdaChain.append(lambda2)
            else:
                lambdaChain.append(lambda1)
    
    return lambdaChain
