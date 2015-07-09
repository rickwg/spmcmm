import numpy as np

def simulate(N):
	'''
	Simulate transition matrix

	Input
	-----
	N : 	int - number of rows

	Returns
	-------
	P : 	ndarray - Transition matrix of size NxN
	'''

	P = np.random.sample(size=(N,N))
	P = P+P.T- np.diag(P.diagonal())
	P = np.array(map(lambda x : (x/sum(x))[::-1],P))
	return P