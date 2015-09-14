def dirichlet_sample(alphas):
    """
    Generate samples from an array of alpha distributions.

    `alphas` must be a numpy array with shape (n, k).
    """
    r = np.random.standard_gamma(alphas)
    r /= r.sum(-1).reshape(-1, 1)
    return r
    
    #y = np.array([np.random.dirichlet(x) for x in alphas])
