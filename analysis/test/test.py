r"""analysis test"""

import numpy as np

#transition matrix for testing
P = np.array(\
	[[0.4, 0.4, 0.2]\
	,[0.4, 0.4, 0.2]\
	,[0.2, 0.2, 0.6]])


eig_values = None
eig_vectors = None
stationnary_distrib = None
time_scales = None
PCCA = None
TPT = None