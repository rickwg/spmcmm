import matplotlib.pyplot as plt


def plot_all2D(data, labels, centers=None):
    """
    plots every possible 2D projection of the Data

    Parameters
    ----------
    data : numpy.array that contains a D dimensional time series
    labels :  numpy.array that contains the label of each point for coloring
    centers: numpy.array, default: None - when centers are handed over they will be also ploted

    -------
    Plot
    """
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
    """
    plots the centers only

    Parameters
    ----------
    centers: numpy.array that contains the centers

    -------
    Plot
    """
    N,D = centers.shape
    k = 0
    for i in range(0, D-1):
        for j in range(i+1, D):
             plt.figure(k)
             k = k+1
             plt.scatter(centers[:, i], centers[:, j])
    plt.show()

def plot_2D(a, b, data, labels, centers=None):
    """
    plots one 2D projection of the Data

    Parameters
    ----------
    a: int that says with column to take for x-axes
    b: int that says with column to take for y-axes
    data : numpy.array that contains a D dimensional time series
    labels :  numpy.array that contains the label of each point for coloring
    centers: numpy.array, default: None - when centers are handed over they will be also ploted

    -------
    Plot
    """
    if centers is not None:
        plt.scatter(centers[:, a], centers[:, b])
    plt.scatter(data[:, a], data[:, b], s=50, c=labels)
    plt.show()