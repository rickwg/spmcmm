import numpy as np
import random
from scipy.spatial import distance

class kMeans:
    """
    Class for K-Means calculation

    Parameters
    ----------
    data : numpy.array that contains a D dimensional time series
    k : int, default: 1 -  the number of clusters to generate
    maxiter: int, default: 1000 - Maximum number of iterations

    Returns
    -------
    centers : numpy.array with the calculated centers
    labels : numpy.array with lables of each point
    """
    def __init__(self, data, k=1, maxiter = 1000):
        N, D = data.shape
        self.centers = np.zeros((k, D))
        self.labels = np.zeros((N, 1))
        self.maxiter = maxiter
        self.k = k
        self.data = data
        self.centerIsSet = False

    def __random_centers(self):
        """
        Compute initial k centers randomly

        Parameters
        ----------

        Returns
        -------
        centers : numpy.array with the initial centers
        """
        kcenters = random.sample(self.data[:, :], self.k)
        kcenters = np.asarray(kcenters)
        return kcenters

    def set_maxiter(self, maxiter):
        self.maxiter = maxiter

    def set_centers(self ,centers):
        self.centers = centers
        self.centerIsSet = True
        self.k, _ = centers.shape

    def get_centers(self):
        return self.centers

    def get_labels(self):
        return self.labels

    def discretize(self):
        """
        the main function to group all samples into k sets

        Parameters
        ----------

        Returns
        -------
        centers : numpy.array with the calculated centers
        labels : numpy.array with lables of each point
        """
        clusterChanged = True
        N,D = self.data.shape
        if self.centerIsSet is False:
            self.centers = self.__random_centers()

        iter = 0
        k = self.k
        data = self.data

        while (clusterChanged & (iter < self.maxiter)):
            clusterChanged = False
            iter = iter+1
            for i in xrange(N):
                #the 1st step of iteration,calculate the distance between each sample and each center
                #for each sample,do the iteration until find the minimal distance to all k centers
                #label each sample using the index of the center which is nearest to it.

                minIdx = distance.cdist(self.centers, [data[i, :]]).argmin()

                if self.labels[i, 0] != minIdx:
                    if self.centerIsSet is False:
                        clusterChanged = True
                    self.labels[i, 0] = minIdx

            #the 2nd step of iteration,calculate the mean value of each labelled group
            #as new center.
            if self.centerIsSet is False:
                for i in range(k):
                    pointsInCluster = data[np.nonzero(self.labels[:, 0] == i)[0]]
                    self.centers[i, :] = np.mean(pointsInCluster, axis=0)