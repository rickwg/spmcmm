import numpy as np
import matplotlib.pyplot as plt

class kMeans:
	def __init__(self, X, k, C=None):
		"""Parameters
		----------
		X :a NxD matrix.
		N :the number of samples.
		D :the dimension of each sample.
		k :the number of centers to choose.
		maxiter :the maximum of iterations."""
		if C is None:
			self.centers = self.__random_centers(X, k)
		else:
			self.centers = C
		N,_ = X.shape
		self.labels = np.zeros((N, 1))
		self.maxiter = 5000
		if C is None:
			self.k = k
		else:
			self.k = C.size

	def __random_centers(self, X, k):
		"""Using the forgy method to chose firstly the initial k centers randomly from N samples.
	    kcenters:a kxD matrix to instore the first k centers."""
		N,D = X.shape
		kcenters = np.zeros((k, D))
		for i in xrange(k):
			idx = int(np.random.uniform(0, N))
			kcenters[i, :] = X[idx, :]
		return kcenters

	def set_maxiter(self ,maxiter):
		self.maxiter = maxiter

	def get_centers(self):
		return self.centers

	def get_labels(self):
		return self.labels

    #the main function to group all samples into k sets.
    #there are 2 iteration steps.
	def discretize(self, X):
		clusterChanged = True
		N,_ = X.shape
		iter = 0

		if C is None:
			while clusterChanged:
				clusterChanged = False

				for i in xrange(N):
					dmin = float("inf")
					minIdx = -1
					#the 1st step of iteration,calculate the distance between each sample and each center
					#for each sample,do the iteration until find the minimal distance to all k centers
					#label each sample using the index of the center which is nearest to it.
					for j in range(self.k):
						dist = np.linalg.norm(self.centers[j, :]-X[i, :])
						if dist < dmin:
							dmin = dist
							minIdx = j

					if self.labels[i, 0] != minIdx:
						clusterChanged = True
						self.labels[i, 0] = minIdx
				#the 2nd step of iteration,calculate the mean value of each labelled group
				#as new center.
				for i in range(self.k):
					pointsInCluster = X[np.nonzero(self.labels[:, 0]== i)[0]]
					self.centers[i, :] = np.mean(pointsInCluster, axis=0)
		else:
			for i in xrange(N):
				dmin = float("inf")
				minIdx = -1
				#the 1st step of iteration,calculate the distance between each sample and each center
				#for each sample,do the iteration until find the minimal distance to all k centers
				#label each sample using the index of the center which is nearest to it.
				for j in range(self.k):
					dist = np.linalg.norm(self.centers[j, :]-X[i, :])
					if dist < dmin:
						dmin = dist
						minIdx = j

				self.labels[i, 0] = minIdx

Y = np.array(((1,5),(3,2),(1,9),(-1,3),(0,5),(7,8)))
#k = 2

#cluster = kMeans(X, k)
#cluster.discretize(X)
#print cluster.get_centers()
#print X
#print cluster.get_labels()


#plt.plot(X[:,0],X[:,1], 'ro')
#plt.plot(cluster.get_centers()[:,0],cluster.get_centers()[:,1], 'bo')
#plt.axis([-2, 10, -2, 10])
#plt.show()


X = np.fromfile('example_1.dat',dtype=float)
X = X.reshape(X.size,1)
k = 3
C = np.zeros((2, 1))
C[1, 0] = 1
#print Y
#print X.reshape(X.size,1).shape

cluster = kMeans(X, k)
cluster.discretize(X)
#print cluster.get_centers()
print cluster.get_centers()
print cluster.get_labels()

#plt.plot(X[:,0], 'ro')
plt.plot(cluster.get_centers()[:,0], 'bo')
#plt.axis([-2, 10, -2, 10])
plt.show()