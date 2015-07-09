from edit import kMeans
import numpy as np
import matplotlib.pyplot as plt


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
k = 10
C = np.zeros((2, 1))
C[1, 0] = 1
#print Y
#print X.reshape(X.size,1).shape

cluster = kMeans(X, k)
cluster.discretize()
#print cluster.get_centers()
print cluster.get_centers()
print cluster.get_labels()

#plt.plot(X[:,0], 'ro')
plt.plot(cluster.get_centers()[:,0], 'bo')
#plt.axis([-2, 10, -2, 10])
plt.show()