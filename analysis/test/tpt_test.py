import numpy as np
from analysis import TPT
import matplotlib.pyplot as plt



# transition matrix
T = np.array([	[0.8, 0.15, 0.05, 0.0, 0.0],\
					[0.1, 0.75, 0.05, 0.05, 0.05],\
					[0.05, 0.1, 0.8, 0.0, 0.05],\
					[0.0, 0.2, 0.0, 0.8, 0.0],\
					[0.0, 0.02, 0.02, 0.0, 0.96]])
A = [0]
B = [4]


tpt = TPT(T,A,B)