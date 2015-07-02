import numpy as np
import estimation as est

import naive_sampling as nsampl


# trans_mat = np.array([[0.5, 0.25, 0.25], [0.5, 0.0, 0.5], [0.25, 0.25, 0.5]], dtype=np.float64)
trans_mat = np.array([[0.25, 0.75], [0.75, 0.25]], dtype=np.float64)
print trans_mat
# chain = nsampl.evolve_chain(0, trans_mat, 20)
chain = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1])
print chain
chain_list = [chain, chain]

# comoute count matrix
c_mat = est.compute_count_matrix(chain, i_tau=2)
c_mat2 = est.compute_count_matrix_list(chain_list)

# estimate transition matrix
t_mat = est.estimate_transition_matrix_naive(c_mat)
t_mat_2 = est.estimate_transition_matrix(c_mat, 10000, 1e-3)

print c_mat
print t_mat
print t_mat_2