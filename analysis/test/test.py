r"""analysis test"""

import numpy as np

#transition matrix for testing
P = np.array(\
	[[0.4, 0.4, 0.2]\
	,[0.4, 0.4, 0.2]\
	,[0.2, 0.2, 0.6]])

eig_values = [0, 0.4, 1]
eig_vectors = 	np.array([[1, -1, 0]\
				,[-1, -1, 2]\
				,[1, 1, 1]])
stationnary_distrib = np.array([1, 1, 1])
stationnary_distrib /= np.sum(stationnary_distrib)
time_scales = map(lambda x: -1./np.log(x) , eig_vectors)
PCCA = None
TPT = None
