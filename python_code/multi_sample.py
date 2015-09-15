# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:50:45 2015

@author: sachin
for the pesky 'object too deep for desired array' errors
"""

import numpy as np

def dirichlet_sample(alphas):
    y = np.array([np.random.dirichlet(x) for x in alphas])
    return y
    
def categorical_sample(prob):
    y = np.array([np.random.multinomial(1,p) for p in prob])
    return y
    