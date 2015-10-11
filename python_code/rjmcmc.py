import numpy as np
from multi_sample import *
from posteriors import *
def posterior_finiteMixture(y, K, extremes, iterations):
    alpha = 5
    N = len(y)
    lambda_ = np.zeros((iterations+1,K))
    pi = np.zeros((iterations+1,K))
    pi[0] = np.random.dirichlet(alpha*np.ones((1,K))[0],1)

    normC = np.diff(np.log(extremes))[0]
    state = 1

    lambda_chain = list()
    lambda_chain.append(np.random.uniform(extremes[0],extremes[1]))
    for i in range(1,iterations):
        if state==1:
            lambda1 = lambda_chain[-1]
            mu1,mu2 = np.random.uniform(0,1,2)
            lambda2 = [lambda1*mu1/(1-mu1), lambda1*(1-mu1)/mu1]
            pi_12 = [mu2,1-mu2]
            lik = np.sum(lambda2*pi_12*np.exp(y*lambda2),1)
            log_joint_lik2 = np.sum(np.log(lik)) \
                             -2*np.log(normC)-np.sum(np.log(lambda2))\
                             -np.log(K)
        else:
             # lambda2 = lambda_[i]
             lambda1 = lambda_chain[-1].prod()
             log_joint_lik1 = (N*np.log(lambda1)-np.sum(lambda1*y)
                            -np.log(normC)-np.log(lambda1)
                            -np.log(K))

        logq = np.log(2)+np.log(lambda1)-np.log(mu1)-np.log(1-mu1)
        alpha_ratio = log_joint_lik2-log_joint_lik1+logq
        A = min(1,np.exp(alpha_ratio))
        if state==2:
            A = min(1,np.exp(-alpha_ratio))

        if A>np.random.uniform(): #accept move
            if state==2:
                state = 1 #switch states
                lambda_chain.append(lambda1)
                # new values of lambda (birth step?)
                lambda1 = q_lambda(y,extremes)
                lambda_chain.append(lambda1)
            else: #state 1
                state = 2 #switch states
                lambda_chain.append(lambda2)
                # new values of lambda (birth step?)
                z = q_z2(y,pi[i-1],lambda_[i-1])
                n_k = sum(z)
                lambda2 = q_lambda2(y,z,n_k,extremes)
                pi[i] = q_pi2(n_k,alpha)
                lambda_chain.append(lambda2)

        else: #keep old value
            lambda_chain.append(lambda_chain[-1])
    
    return lambda_chain
