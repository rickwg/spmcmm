#!/usr/bin/python

import sys
import numpy as np
import tools

class MarkovModel():
    """
    MCMM Analysis class.
    """
    def __init__(self, T, lagtime = 1., test=False):
        if not test:
            self.T = T
            try:
                assert type(self.T) == np.ndarray
            except:
                raise TypeError('Matrix is not a ndarray')
            if self.is_transition_matrix() and self.is_irreducible():
                self.lagtime = float(lagtime)
                self.timeScales = None
                self.statDist = None
                self.eigVal = None
                self.eigVec = None
                self.pcca = None
        else:
            self.T = T
            self.lagtime = float(lagtime)
            self.timeScales = None
            self.statDist = None
            self.eigVal = None
            self.eigVec = None
            self.pcca = None

    def is_transition_matrix(self):
        """
        Check if the given matrix is a transition matrix
        """
        if len(self.T.shape)  != 2 :    # check matrix is 2D
            raise Exception('not a 2D matrix')
            return False
        if self.T.shape[0] != self.T.shape[1]:  # check matrix is square
            raise Exception('matrix is not square')
            return False    
        if np.any(self.T<0):
            raise Exception('matrix terms are not all positive')
            return False
        if (np.sum(self.T, axis=1) != 1.).all():
            raise Exception('sum of raw term is not 1') # Sum of rows is equal to 1
            return False
        else:
            return True 

    def is_irreducible(self):
        """
        Check if a matrix is irreducible
        """
        if len(tools.communication_classes(self.T)) != 1 :
            raise Exception('matrix is not irreducible')
            return False
        else:
            return True 

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
        eigVal, eigVec = np.linalg.eig(self.T)

        #sort decreasingly the eigen vectors with respect to the eigen values
        self.eigVal, self.eigVec = zip(*sorted(zip(*[eigVal,eigVec]), reverse=True))
        return self.eigVal, self.eigVec

    def statDistribution(self):
        """
        Compute and return the statistical distribution of a 
        transition matrix T
        
        Returns
        -------
        statDist : (N) list
        vector of the stationnary distribution 
        """
        eigVal, eigVec = np.linalg.eig(self.T)
        self.statDist = eigVec[np.where(np.isclose(eigVal,1))].flatten()
        return self.statDist

    def timescales(self, lagtime = 1.0):
        """
        Compute and return the time scales of a transition matrix T
        lagtime = 1. default

        Returns
        -------
        timeScales
        """
        if self.lagtime:
            lagt = self.lagtime
        else:
            lagt = lagtime

        if self.eigVal == None:
            self.eigenVectors()

        realEigVal = np.real(self.eigVal)
        self.timeScales = np.zeros(realEigVal.shape)

        for j in range(len(realEigVal)):
            # Take care : ZeroDivisionError
            if np.isclose((realEigVal[j]-1.)**2,0):
                self.timeScales[j] = np.inf
            else:
                self.timeScales[j] = -lagt / np.log(np.absolute(realEigVal[j]))
        return self.timeScales

    def PCCA(self, m):
        """Use pyemma pcca """      
        from pyemma.msm.analysis import pcca
        self.pcca = pcca(self.T, m)
        return self.pcca


class TPT():
    def __init__(self, T, a, b):
        MarkovModel(T).is_transition_matrix() # Let's raise the exceptions in the is_transition_matrix() function
        MarkovModel(T).is_irreducible() # Let's raise the exceptions in the is_irreducible() function
        
        try:    # The two subsets A & B checked for transition
            assert type(a) == list
            assert type(b) == list
            ax,bx = set(a),set(b)
            assert len(ax.intersection(bx)) == 0
        except:
            raise Error('sets are not disjoints')
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
        self.flux = None

    def forwardCommittor(self):
        """
        Compute and return the forward committor
        Conditions ruling the forward committor:
                            left part : G           right part : d
        if i not in AUB :   sum_{j} L_{ij}*q_{j}    = 0
        if i in A :         q_{i}                   = 0
        if i in B :         q_{i}                   = 1
        """
        from scipy.linalg import solve
        n = self.T.shape[0]
        # L = generator matrix
        L = self.T - np.eye(n) # np.eye = identity matrix
        
        # left part of the equation
        G = np.eye(n)
        for i in range(n):
            if i not in self.a and i not in self.b:
                G[i] = L[i]
        
        #right part of the equation
        d = np.zeros((n,1))
        for i in range(n):
            if i in self.b:
                d[i] = 1

        #solve the equation
        self.forwardCommit = solve(G,d).flatten()
        return self.forwardCommit

    def backwardCommittor(self):
        """
        Compute and return the backward committor
        Conditions ruling the backward committor:
                            left part : G           right part : d
        if i not in AUB :   sum_{j} L_{ij}*q_{j}    = 0
        if i in A :         q_{i}                   = 1
        if i in B :         q_{i}                   = 0
        """
        from scipy.linalg import solve
        n = self.T.shape[0]
        # L = generator matrix
        L = self.T - np.eye(n) # np.eye = identity matrix
        
        # left part of the equation
        G = np.eye(n)
        for i in range(n):
            if i not in self.a and i not in self.b:
                G[i] = L[i]
        
        #right part of the equation
        d = np.zeros((n,1))
        for i in range(n):
            if i in self.a:
                d[i] = 1

        #solve the equation
        self.backwardCommit = solve(G,d).flatten()
        return self.backwardCommit

    def probabilityCurrent(self):
        """
        Compute and return the probability current
        """

        # create a zero diagonal that will put to zero all the i==j cases
        probaCurrent_ii = np.ones(self.T.shape)-np.eye(self.T.shape[0])
        if self.forwardCommit == None:
            self.forwardCommittor()
        if self.backwardCommit == None:
            self.backwardCommittor()
        if self.statDist == None:
            self.statDistribution = MarkovModel(self.T)
        
        # compute stationnary distribution
        self.statDist = MarkovModel(self.T).statDistribution()
        self.probaCurrent = ((((self.T*self.forwardCommit).T*self.backwardCommit)*self.statDist)*probaCurrent_ii).T
        return self.probaCurrent

    def effectiveProbabilityCurrent(self):
        """
        Compute and return the effective probability current
        """
        if self.probaCurrent == None:
            self.probabilityCurrent()
        # initialise effective probability current
        self.effectiveProbaCurrent = np.zeros(self.T.shape)
        # indices of where probability current [i,j] > probability current [j,i]
        indSup = zip(*np.where(self.probaCurrent - np.array(self.probaCurrent).T >= 0))
        # indices of where probability current [i,j] < probability current [j,i]
        indInf = zip(*np.where(self.probaCurrent - np.array(self.probaCurrent).T < 0))
        for i in indSup:
            self.effectiveProbaCurrent[i] = self.probaCurrent[i] - self.probaCurrent[i[::-1]]
            self.effectiveProbaCurrent[i[::-1]] = 0
        for i in indInf:
            self.effectiveProbaCurrent[i] = 0
            self.effectiveProbaCurrent[i[::-1]] = self.probaCurrent[i[::-1]] - self.probaCurrent[i]
        return self.effectiveProbaCurrent

    def filux(self):
        """
        Compute and return the flux = "Average total number of trajectories from A to B per time unit"
        """
        if self.effectiveProbaCurrent == None:
            self.effectiveProbabilityCurrent()
        self.flux = np.sum([np.sum(self.effectiveProbaCurrent[x]) for x in self.a])
        return self.flux

    def transitionrate(self):
        """
        Compute and return the transition rate = (Flux)/(Total number of trajectories going forward from A)
        """
        if self.flux == None:
            self.filux()
        self.transitionRate = self.flux/np.sum(self.statDist*self.backwardCommit)
        return self.transitionRate

    def meanfirstpassagetime(self):
        """
        Compute and return the mean first passage time = inverse of transition rate
        """
        if self.transitionRate == None:
            self.transitionrate()
        self.meanFirstPassageTime = self.transitionRate**(-1)
        return self.meanFirstPassageTime

    def minCurrent(self,w):
        """
        Compute and return the min-current (capacity) of a pathway w
        """
        from itertools import product
        if self.effectiveProbaCurrent == None:
            self.effectiveProbabilityCurrent()
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
