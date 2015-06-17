r"""algebra - analysis"""

import numpy as np

def eigen_vectors(P):
	"""
	Compute the eigen values and eigen vectors 
	of a transition matrix and sort them from the biggest 
	eigen value to the smallest.
	
	Parameters
	----------
	P : (NxN) ndarray
	transition matrix
	
	Returns
	-------
	(eig_val, eig_vec)
	
	eig_val : (N) list of eigen values

	eig_vec : (NxN) ndarray
	array of eigen vectors 

	"""
	#take eigen values & eigen vectors from P
	eig_val, eig_vect = np.linalg.eig(P)

	#sort decreasingly the eigen vectors with respect to the eigen values
	eig_val, eig_vect = zip(*sorted(zip(*[eig_val,eig_vect]), reverse=True))

	return eig_val, eig_vect


def stat_distribution(P):
	"""
	Compute the statistical distribution of a 
	transition matrix
	
	Parameters
	----------
	P : (NxN) ndarray
	transition matrix
	
	Returns
	-------
	stat_dist : (N) list
	vector of the stationnary distribution 

	"""
	eig_val, eig_vect = np.linalg.eig(P)
	stat_dist = eig_vect[np.where(eig_val==1)]
	
	return stat_dist