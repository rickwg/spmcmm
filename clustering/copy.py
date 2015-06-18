__author__ = 'Alo'
import  numpy as np

class kMeans:
    def __init__(self,X,k):
        """X is a N*D matrix:N is the number of samples,D is the dimension of sample"""
        self.centers=self._random_centers(X,k)
        N,_=X.shape
        self.labels=np.zeros((N,1))
        self.maxiter=5000
        self.k=k

    def _random_centers(self,X,k):
        N,D=X.shape
        kcenters=np.zeros((k,D))
        for i in range(k):
            idx=int(np.random.uniform(0,N))
            kcenters[i,:]=X[idx,:]
        return  kcenters

    def set_maxiter(self,maxiter):
        self.maxiter=maxiter

    def get_centers(self):
        return self.centers

    def get_labels(self):
        return self.labels

    def run(self,X):
        clusterChanged=True
        N,_=X.shape
        iter =0

        while clusterChanged:
            clusterChanged=False

            for i in xrange(N):
                minDist=float("inf")
                minIdx=-1

                for j in xrange(self.k):
                    dist=np.linalg.norm(self.centers[j,:],X[i,:])
                    if dist<minDist:
                        minIdx=j

                    if self.labels[i,0]!=minIdx:
                        clusterChanged=True
                        self.labels[i,0]=minIdx

                for i in range(self.k):
                    pointsInCluster=X[np.nonzero(self.labels[:,0]==i)[0]]
                    self.centers[i, :] = np.mean(pointsInCluster, axis = 0)

k_means = kMeans(X,3)
k_means.run(X)
print k_means.get_centers()


