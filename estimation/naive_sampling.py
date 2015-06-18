
import numpy as np

def tower_sample(dist):
    cdf = np.cumsum(dist)
    rnd = np.random.rand() * cdf[-1]
    ind = (cdf > rnd)
    idx = np.where(ind == True)
    return np.min(idx)

def evolve_chain(x, trans_mat, length):
    chain = np.zeros(length, dtype=np.intc)
    chain[0] = x
    for i in xrange(1, length):
        chain[i] = tower_sample(trans_mat[chain[i-1]])
    return chain
