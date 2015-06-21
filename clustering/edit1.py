__author__ = 'Alo'
import numpy as np


class kMeans:
	def __init__(self,X,k):
		"""Parameters:
		X :a NxD matrix.
		N :the number of samples.
		D :the dimension of each sample.
		k:the number of centers to choose.
		maxiter:the maximum of iterations."""
		self.centers = self.__random_centers(X,k)
		N,_ = X.shape
		self.labels = np.zeros((N,1))
		self.maxiter = 5000
		self.k = k

	def __random_centers(self,X,k):
		"""Using the forgy method to chose firstly the initial k centers randomly from N samples.
	    kcenters:a kxD matrix to instore the first k centers."""
		N,D = X.shape
		kcenters = np.zeros((k,D))
		for i in xrange(k):
			idx = int(np.random.uniform(0,N))
			kcenters[i,:] = X[idx,:]
		return kcenters

	def set_maxiter(self,maxiter):
		self.maxiter = maxiter

	def get_centers(self):
		return self.centers

	def get_labels(self):
		return self.labels

    #the main function to group all samples in k sets.
    #there are 2 iteration steps.
	def discretize(self,X):
		clusterChanged = True
		N,_ = X.shape
		iter = 0

		while clusterChanged:
			clusterChanged = False
			iter = iter + 1
			if iter > self.maxiter:
				break

			for i in xrange(N):
				dmin = float("inf")
				minIdx = -1

            #the 1st step of iteration,calculate the distance between each sample and each center
            #for each sample,do the iteration until find the minimal distance to all k centers
            #label each sample using the index of the center which is nearest to it.
				for j in range(self.k):
					dist = np.linalg.norm(self.centers[j,:],X[i,:])
					if dist < dmin:
						dmin = dist
						minIdx = j

				if self.labels[i,0] != minIdx:
					clusterChanged = True
					self.labels[i,0] = minIdx

            #the 2nd step of iteration,calculate the mean value of each labelled group
            #as new center.
			for i in range(self.k):
				pointsInCluster = X[np.nonzero(self.labels[:, 0]== i)[0]]
				self.centers[i, :] = np.mean(pointsInCluster, axis = 0)


Cluster = kMeans(X,3)
Cluster.run(X)
print Cluster.get_centers()



