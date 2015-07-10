import analysis as an
import numpy as np

T = np.array(
[[ 0.22236564,0.15408099,0.2150307,0.17571315,0.23280952],
 [ 0.16133225,0.173507,0.27971911, 0.23056001 ,0.15488163],
 [ 0.1340729 , 0.1665676,  0.30992098, 0.27281507, 0.11662346],
 [ 0.13024635, 0.16321992, 0.32433151, 0.27543921, 0.10676302],
 [ 0.24272049, 0.15421744, 0.19500741, 0.15016379, 0.25789087]]
	)


''' MARKOV MODEL '''
#Create Markov Model Object
mm = an.MarkovModel(T)

#compute eigen vectors & eigen values
mm.eigenVectors()

#compute stationnary distribution
mm.statDistribution()

#compute time scales
mm.timescales()

#compute PCCA
mm.PCCA()


''' Transition Path Theory '''

#Create TPT Object
tpt = an.TPT(T)

# Compute forward & backward committors
tpt.forwardCommittor()
tpt.backwardCommittor()

# compute proba current
tpt.probabilityCurrent()

# compute effective proba current
tpt.effectiveProbabilityCurrent()

# compute flux
tpt.filux()

# compute transition rate
tpt.transitionrate()

# compute mean first passage time
tpt.meanfirstpassagetime()