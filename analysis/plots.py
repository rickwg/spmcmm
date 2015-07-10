import numpy as np
from analysis import MarkovModel
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm

# transition matrix for test
T = np.array([  [0.8, 0.15, 0.05, 0.0, 0.0],\
                    [0.1, 0.75, 0.05, 0.05, 0.05],\
                    [0.05, 0.1, 0.8, 0.0, 0.05],\
                    [0.0, 0.2, 0.0, 0.8, 0.0],\
                    [0.0, 0.02, 0.02, 0.0, 0.96]])


# Plotting stuff
colors =    [['lightsteelblue', 'steelblue'], ['palegoldenrod', 'gold'], \
            ['lightgreen', 'mediumseagreen'], ['pink', 'palevioletred'], \
            ['silver','gray'], ['bisque', 'darksalmon'], ['thistle','mediumorchid'], ['paleturquoise', 'mediumturquoise']]
colors = colors*100

font = fm.FontProperties(size=14)
font.set_family('serif')
font.set_weight('light')
# End plotting stuff

def plotHeatMap(T):
    plt.pcolor(T, cmap ='PuRd')
    plt.colorbar()

def plotEigVectors(eigVect):
    N = len(eigVect)
    bins = 1000                     # number of bins
    x = np.arange(0,N, 1./bins)     # x axis (arbitrary unit)

    f, ax = plt.subplots(N, sharex=True)
    
    # add space between subplots
    f.subplots_adjust(hspace=.5)

    ax[0].set_title('Eigen Vectors', fontproperties = font)

    minXLim = 0
    maxXLim = len(eigVect[0])
    minYLim = np.min(eigVect)-0.2*abs(np.min(eigVect))
    maxYLim = np.max(eigVect)*1.2
    for i in xrange(N):
        # duplicate each value bins times
        y = [v for v in eigVect[i] for _ in range(bins)]

        ax[i].fill_between(x,0,y, facecolor=colors[i][0], alpha=0.5,linewidth=0.5)
        ax[i].plot(x,y,linewidth=1.5, color=colors[i][1])
        ax[i].axis((minXLim,maxXLim,minYLim,maxYLim))
    plt.show()


def plotEigValues(eigValues):
    N = len(eigValues)
    
    f, ax = plt.subplots(1)
    ax.set_title('Eigen Values', fontproperties = font)

    for j in xrange(N):
        markerline, stemlines, baseline = ax.stem([j], [eigValues[j]], '-.')
        plt.setp(stemlines, linewidth=2, color=zip(*colors)[1][j])
        plt.setp(markerline, markerfacecolor='lightgrey')
    plt.margins(0.1, 0.01)
    plt.show()