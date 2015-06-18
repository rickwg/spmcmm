
import numpy as np

def compute_count_matrix(i_chain):
    n_states = i_chain.max() + 1
    count_matrix = np.zeros((n_states, n_states), dtype=np.intc)

    for i in xrange(1, i_chain.shape[0]):
        count_matrix[i_chain[i-1], i_chain[i]] += 1
    return count_matrix

def estimate_transition_matrix(i_count_mat):
    est_trans_mat = np.zeros(i_count_mat.shape, dtype=np.float64)

    for i in xrange(est_trans_mat.shape[0]):
        row = i_count_mat[i, :]
        est_trans_mat[i, :] = np.float64(row) / np.sum(np.float64(row))
    return est_trans_mat


