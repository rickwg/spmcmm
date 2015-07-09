from edit import kMeans
import numpy as np
import matplotlib.pyplot as plt
from ClusterPlot import *

Y = np.array(((1,0),(1,2),(2,1),(15,4),(13,2),(21,4)))
centerY = np.array(((0,0),(20,5)))
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

dat = np.genfromtxt('example_2.dat', delimiter=' ')


X = np.fromfile('example_2.dat',dtype=float)
X = X.reshape(X.size, 1)
#k = 3
#C = np.zeros((2, 1))
#C[1, 0] = 1
#print Y
#print X.reshape(X.size,1).shape

#cluster = kMeans(X, k)
#cluster.discretize()
#print cluster.get_centers()
#print cluster.get_centers()
#print cluster.get_labels()

#plt.plot(X[:,0], 'ro')
#plt.plot(cluster.get_centers()[:,0], 'bo')
#plt.axis([-2, 10, -2, 10])
#plt.show()


testOneD = np.array(((1),(1.1),(0.9),(130),(130.1),(129.9)))
testOneD = testOneD.reshape(testOneD.size,1)
k = 2

cluster = kMeans(Y, k)
cluster.set_centers(centerY)
cluster.discretize()

print cluster.get_centers()
print cluster.get_labels()

colormap = np.array(['r', 'g', 'b'])


plot_all2D(Y, cluster.get_labels(), cluster.get_centers())
#plot_center(cluster.get_centers())
