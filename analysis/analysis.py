#!/usr/bin/python

import numpy as np
from tools import communication_classes, depth_first_search

class MarkovModel():
	def __init__(self, T, lagtime = 1.):

		self.T = T
		try:
			assert self.is_transition_matrix()
			assert type(self.T) == np.ndarray
		except Exception, e:
			raise e
		
		self.lagtime = float(lagtime)
		self.timeScales = None
		self.statDist = None
		self.eigVal = None
		self.eigVec = None
		self.pcca = None

	def is_transition_matrix(self):
		'''
		Check if the given matrix is a transition matrix
		'''
		try:
			assert len(self.T.shape) == 2				# 2D matrix
			assert self.T.shape[0] == self.T.shape[1]	# Square matrix
			assert (self.T >= 0).all()					# All positive
			assert np.sum(P, axis=1) == [1]*P.shape[0]	# Sum of rows = 1
			return True
		except:
			return False

	def is_irreducible(self):
		'''
		Check if a matrix is irreducible
		'''
		return len(communication_classes(self.T)) == 1

	def eigenVectors(self):
		'''
		Compute the eigen values and eigen vectors 
		of a transition matrix P and sort them from the biggest 
		eigen value to the smallest.

		Returns
		-------
		(eig_val, eig_vec)
		
		eig_val : (N) list of eigen values
		eig_vec : (NxN) ndarray
		array of eigen vectors 
		'''
		#take eigen values & eigen vectors from P
		eigVal, eigVec = np.linalg.eig(self.T)

		#sort decreasingly the eigen vectors with respect to the eigen values
		self.eigVal, self.eigVec = zip(*sorted(zip(*[eigVal,eigVec]), reverse=True))

		return self.eigVal, self.eigVec


	def statDistribution(self):
		'''
		Compute and return the statistical distribution of a 
		transition matrix T
		
		Returns
		-------
		statDist : (N) list
		vector of the stationnary distribution 
		'''
		eigVal, eigVec = np.linalg.eig(self.T)
		self.statDist = eigVec[np.where(eigVal==1)]
		return self.statDist

	def timescales(self):
		'''
		Compute and return the time scales of a transition matrix T
		lagtime = 1. default

		Returns
		-------
		timeScales
		'''
		realEigVal = np.real(self.eigVal)
		self.timeScales = np.zeros(realEigVal.shape)

		for j in range(len(realEigVal)):
			# Take care : ZeroDivisionError
			if np.isclose(realEigVal[j]-1.)**2,0):
				self.timeScales[j] = np.inf
			else:
				self.timeScales[j] = -self.lagtime / np.log(np.absolute(realEigVal[j]))
		return self.timeScales


	def pcca(self, m):
		'''Use pyemma pcca '''
		import pcca as pyemma_pcca
		self.pcca = pyemma_pcca(self.T, m)
		return self.pcca


class TPT():
	def __init__(self, T, a, b):
		try: 
			assert is_transition_matrix(T)
			# A & B disjunct
			assert len(a.intersection(b)) == 0
			assert type(a) == list
			assert type(b) == list
		except Exception, e:
			# Let's raise the exceptions in the is_transition_matrix() function
			raise e
		# The two subsets A & B checked for transition
		self.T = T
		self.a = a
		self.b = b

		# Properties
		self.forwardCommit = None
		self.backwardCommit = None
		self.probaCurrent = None
		self.effectiveProbaCurrent = None
		self.transitionRate = None
		self.meanFirstPassageTime = None
		self.statDist = None

	def forwardCommittor(self):
		from scipy.linalg import solve
		'''
		Compute and return the forward committor
		Conditions ruling the forward committor:
							left part : G			right part : d
		if i not in AUB : 	sum_{j} L_{ij}*q_{j} 	= 0
		if i in A : 		q_{i} 					= 0
		if i in B : 		q_{i} 					= 1
		'''
		n = self.T.shape[0]
		# L = generator matrix
		L = self.T - np.eye(n) # np.eye = identity matrix
		
		# left part of the equation
		G = np.eye(n)
		for i in range(n):
			if i not in self.a and i not in self.b:
				W[i] = L[i]
		
		#right part of the equation
		d = np.zeros(1,n)
		for i in range(n):
			if i in self.b:
				d[i] = 1

		#solve the equation
		self.forwardCommit = solve(G,d)
		return self.forwardCommit

	def backwardCommittor(self):
		from scipy.linalg import solve
		'''
		Compute and return the backward committor
		Conditions ruling the backward committor:
							left part : G			right part : d
		if i not in AUB : 	sum_{j} L_{ij}*q_{j} 	= 0
		if i in A : 		q_{i} 					= 1
		if i in B : 		q_{i} 					= 0
		'''
		n = self.T.shape[0]
		# L = generator matrix
		L = self.T - np.eye(n) # np.eye = identity matrix
		
		# left part of the equation
		G = np.eye(n)
		for i in range(n):
			if i not in self.a and i not in self.b:
				W[i] = L[i]
		
		#right part of the equation
		d = np.zeros(1,n)
		for i in range(n):
			if i in self.b:
				d[i] = 1

		#solve the equation
		self.backwardCommit = solve(G,d)
		return self.backwardCommit

	def probabilityCurrent(self):
		'''
		Compute and return the probability current
		'''

		# create a zero diagonal that will put to zero all the i==j cases
		probaCurrent_ii = np.ones(self.T.shape)-np.eye(self.T.shape)
		
		# compute stationnary distribution
		self.statDist = MarkovModel(self.T).statDistribution()

		picucu = np.kron(self.statDist*self.backwardCommit, self.forwardCommit)
		self.probaCurrent = np.dot(picucu.reshape(self.T.shape)*self.T, probaCurrent_ii)
		return self.probaCurrent

	def effectiveProbabilityCurrent(self):
		'''
		Compute and return the effective probability current
		'''

		# initialise effective probability current
		self.effectiveProbaCurrent = np.zeros(self.T.shape)

		# indices of where probability current [i,j] > probability current [j,i]
		indSup = np.where(self.probaCurrent > np.array(self.probaCurrent).T)
		# indices of where probability current [i,j] < probability current [j,i]
		indInf = np.where(self.probaCurrent < np.array(self.probaCurrent).T)

		for i in indSup:
			self.effectiveProbaCurrent[i] = self.probaCurrent[i] - self.probaCurrent[i[::-1]]
			self.effectiveProbaCurrent[i[::-1]] = 0
		for i in indInf:
			self.effectiveProbaCurrent[i] = 0
			self.effectiveProbaCurrent[i[::-1]] = self.probaCurrent[i[::-1]] - self.probaCurrent[i]
		return self.effectiveProbaCurrent

	def flux(self):
		'''
		Compute and return the flux = "Average total number of trajectories from A to B per time unit"
		'''
		self.flux = np.sum([np.sum(self.probabilityCurrent[x]) for x in self.a])
		return self.flux

	def transitionRate(self):
		'''
		Compute and return the transition rate = (Flux)/(Total number of trajectories going forward from A)
		'''
		self.transitionRate = self.flux/np.sum(self.statDist*self.backwardCommit)
		return self.transitionRate

	def meanFirstPassageTime(self):
		'''
		Compute and return the mean first passage time = inverse of transition rate
		'''
		self.meanFirstPassageTime = self.transitionRate**(-1)
		return self.meanFirstPassageTime

	def minCurrent(self,w):
		'''
		Compute and return the min-current (capacity) of a pathway w
		'''
		from itertools import product
		try:
			assert w[0] in self.a
			assert w[-1] in self.b
			for j in w[1:-2]:
				assert w[j] not in np.union1d(self.a, self.b)
		except Exception, e:
			raise TypeError('w is not a pathway')
		# all possible indices for all i,j in w
		indices = product(w, repeat = 2)
		return np.min([self.effectiveProbaCurrent[i] for i in indices])
