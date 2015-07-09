import numpy as np
import random

class kMeans:
    def __init__(self, data, k=1):
        """Parameters
        ----------
        X :a NxD matrix.
        N :the number of samples.
        D :the dimension of each sample.
        k :the number of centers to choose.
        maxiter :the maximum of iterations."""
        #self.centers = self.__random_centers(X, k)

        N,D = data.shape
        self.centers = np.zeros((k, D))
        self.labels = np.zeros((N, 1))
        self.maxiter = 5000
        self.k = k
        self.data = data
        self.centerIsSet = False


    def __random_centers(self):
        """Using the forgy method to chose firstly the initial k centers randomly from N samples.
        kcenters:a kxD matrix to instore the first k centers."""
        kcenters = random.sample(self.data[:, :], self.k)
        kcenters = np.asarray(kcenters)
        print kcenters
        return kcenters

    def set_maxiter(self ,maxiter):
        self.maxiter = maxiter

    def set_centers(self ,centers):
        self.centers = centers
        self.centerIsSet = True
        _, self.k = centers.shape

    def get_centers(self):
        return self.centers

    def get_labels(self):
        return self.labels

    #the main function to group all samples into k sets.
    #there are 2 iteration steps.
    def discretize(self):
        clusterChanged = True
        N,D = self.data.shape
        if self.centerIsSet is False:
             #self.centers = np.asarray([self.data[np.random.randint(0, len(self.data))] for _ in xrange(k)], dtype=np.float32)
		     self.centers = self.__random_centers()

        iter = 0

        while clusterChanged:
            clusterChanged = False

            for i in xrange(N):
                dmin = float("inf")
                minIdx = -1
                #the 1st step of iteration,calculate the distance between each sample and each center
                #for each sample,do the iteration until find the minimal distance to all k centers
                #label each sample using the index of the center which is nearest to it.
                for j in range(self.k):
                    dist = np.linalg.norm(self.centers[j, :]-self.data[i, :])
                    if dist < dmin:
                        dmin = dist
                        minIdx = j

                if self.labels[i, 0] != minIdx:
                    if self.centerIsSet is False:
                        clusterChanged = True
                    self.labels[i, 0] = minIdx
            #the 2nd step of iteration,calculate the mean value of each labelled group
            #as new center.
            if self.centerIsSet is False:
                for i in range(self.k):
                    pointsInCluster = self.data[np.nonzero(self.labels[:, 0] == i)[0]]
                    self.centers[i, :] = np.mean(pointsInCluster, axis=0)