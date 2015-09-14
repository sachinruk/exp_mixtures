def normalise(logp):
    """
    returns exp(logp)/sum(exp(logp)) without numerical 
    problems for a NxD matrix
    """"
    max_logp=np.amax(logp,1);
    logp=logp-max_logp;
    p=np.exp(logp);
    p=p/np.sum(p,1);

    return p
