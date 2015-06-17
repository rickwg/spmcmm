r"""pcca methods - analysis"""

def is_transition_matrix(P):
	"""
	Compute the pcca of a transition matrix
	
	Parameters
	----------
	P : (nxn) ndarray
	transition matrix

	Returns
	-------
	bool : 	True if the matrix is a transition matrix
			False either
	"""
	try:
		#sum of all the lines must be one
		assert np.sum(P, axis=1) == [1]*P.shape[0]
		return True
	except:
		return False

def pcca(P, n_clusters):
	"""
	Compute the pcca of a transition matrix
	
	Parameters
	----------
	P : (nxn) ndarray
	transition matrix
	
	n_clusters : m
	number of clusters to group

	Returns
	-------
	Chi : (nxm) ndarray
	membership matrix containing the assignment probabilities of each microstate to
	belong to a macrostate
	"""

	return Chi
