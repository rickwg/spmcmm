import matplotlib.pyplot as plt


def plot_all2D(data, labels, centers=None):
    N,D = data.shape
    k = 0
    for i in range(0, D-1):
        for j in range(i+1, D):
             plt.figure(k)
             k = k+1
             if centers is not None:
                 plt.scatter(centers[:, i], centers[:, j])
             plt.scatter(data[:, i], data[:, j], s=50, c=labels)
    plt.show()


def plot_center(centers):
    N,D = centers.shape
    k = 0
    for i in range(0, D-1):
        for j in range(i+1, D):
             plt.figure(k)
             k = k+1
             plt.scatter(centers[:, i], centers[:, j])
    plt.show()