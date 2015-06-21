#!/usr/bin/python

import numpy as np
import algebra as al
import pcca
from tools import is_irreducible, communication_classes, depth_first_search, is_transition_matrix


class MarkovModel():

	def eigenVectors(self):
		"""
		Compute the eigen values and eigen vectors 
		of a transition matrix P and sort them from the biggest 
		eigen value to the smallest.

		Returns
		-------
		(eig_val, eig_vec)
		
		eig_val : (N) list of eigen values
		eig_vec : (NxN) ndarray
		array of eigen vectors 
		"""
		#take eigen values & eigen vectors from P
		eig_val, eig_vect = np.linalg.eig(self.T)

		#sort decreasingly the eigen vectors with respect to the eigen values
		eig_val, eig_vect = zip(*sorted(zip(*[eig_val,eig_vect]), reverse=True))

		return eig_val, eig_vect


	def statDistribution(self):
		"""
		Compute the statistical distribution of a 
		transition matrix P
		
		Returns
		-------
		stat_dist : (N) list
		vector of the stationnary distribution 
		"""
		eig_val, eig_vect = np.linalg.eig(self.T)
		stat_dist = eig_vect[np.where(eig_val==1)]
		return stat_dist

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
		self.meanPassageTime = None

	def forwardCommittor(self):
		from scipy.linalg import solve
		'''
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