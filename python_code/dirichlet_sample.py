import numpy as np
def dirichlet_sample1(alphas):
    """
    Generate samples from an array of alpha distributions.

    `alphas` must be a numpy array with shape (n, k).
    """
    
    r = np.random.standard_gamma(alphas)
    r /= r.sum(-1).reshape(-1, 1)
    return r
    

def dirichlet_sample2(alphas):
    y = np.array([np.random.dirichlet(x) for x in alphas])
    return y
    
    
def wrapper(func, *args):
    def wrapped():
        return func(*args)
    return wrapped

def print_timings():
    a=np.reshape(np.random.exponential(10,10),(5,2))
    wrapped = wrapper(dirichlet_sample1, a)
    print timeit.timeit(wrapped, number=10000)    
    wrapped = wrapper(dirichlet_sample2, a)
    print timeit.timeit(wrapped, number=10000)    
    #dirichlet sample2 is FASTER