import numpy as np
import estimation.estimation as est
import scipy.linalg as lina

import naive_sampling as nsampl


trans_mat = np.array([[0.5, 0.25, 0.25], [0.5, 0.0, 0.5], [0.25, 0.25, 0.5]], dtype=np.float64)
# trans_mat = np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float64)
print trans_mat
chain = nsampl.evolve_chain(0, trans_mat, 20000)
# chain = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0])
print chain
chain_list = [chain, chain]

# comoute count matrix
c_mat = est.compute_count_matrix(chain, i_tau=1)
c_mat2 = est.compute_count_matrix_list(chain_list)

# estimate transition matrix
t_mat = est.estimate_transition_matrix(c_mat, 100000, 1e-3)

eval, evec_left, evec_right = lina.eig(t_mat, left=True)

# sort eigenvalues descending
idx = eval.argsort()[::-1]
stationary_dist = evec_left[:, idx[0]].real

S = np.dot(np.diag(stationary_dist), t_mat)

print c_mat
print t_mat
print np.sum(t_mat, axis=1)
print np.linalg.norm(t_mat - trans_mat, ord='fro') / np.linalg.norm(trans_mat, ord='fro')
print S