import numpy as np
import scipy.linalg as lina


def compute_count_matrix(i_chain, i_tau=1):
    """
    Compute count matrix of a markov chain

    Parameters
    ----------
    i_chain : numpy.array that contains a discrete markov chain


    Returns
    -------
    count_matrix : numpy.array

    """

    n_states = i_chain.max() + 1
    count_matrix = np.zeros((n_states, n_states), dtype=np.intc)

    for i in xrange(i_tau, i_chain.shape[0]):
        count_matrix[i_chain[i - i_tau], i_chain[i]] += 1
    return count_matrix

def compute_count_matrix_list(i_chain_list, i_tau=1):
    """
    Compute count matrix of a list of markov chains

    Parameters
    ----------
    i_chain_list : numpy.array that contains a discrete markov chain

    Returns
    -------
    count_matrix : numpy.array

    """
    n_states = i_chain_list.max() + 1
    count_matrix = np.zeros((n_states, n_states), dtype=np.intc)

    for i in xrange(i_chain_list.shape[0]):
        count_matrix += compute_count_matrix(i_chain_list[i, :], i_tau)
    return count_matrix


def estimate_transition_matrix_naive(i_count_mat):
    """
    Estimate transition matrix

    Parameters
    ----------
    i_count_mat : numpy.array

    Returns
    -------
    est_trans_mat : numpy.array

    """
    est_trans_mat = np.zeros(i_count_mat.shape, dtype=np.float64)

    for i in xrange(est_trans_mat.shape[0]):
        row = i_count_mat[i, :]
        est_trans_mat[i, :] = np.float64(row) / np.sum(np.float64(row))
    return est_trans_mat

def compute_init_value(i_count_mat):
    """
    Helper function. Computes a initial value for estimation of
    transition matrix

    Parameters
    ----------
    i_count_mat : numpy.array

    Returns
    -------
    est_trans_mat : numpy.array

    """
    # compute naive transition matrix T_ij = c_ij / sum_j c_ij
    # and stationary distribution
    T_temp = estimate_transition_matrix_naive(i_count_mat)
    eval, evec_left, evec_right = lina.eig(T_temp, left=True)

    # sort eigenvalues descending
    idx = eval.argsort()[::-1]
    stationary_dist = evec_left[:, idx[0]].real

    # compute initial value x0
    x0 = np.dot(np.diag(stationary_dist), T_temp)
    del T_temp

    return x0

def estimate_transition_matrix(i_count_mat, i_max_iter, i_tol):
    """
    Estimate transition matrix with detailed balance condition,
    i.e for reversible transition matrices

    Parameters
    ----------
    i_count_mat : numpy.array, count matrix
    i_max_iter: float, max number of iterations
    i_tol: float, tolerance

    Returns
    -------
    est_trans_mat : numpy.array

    """
    n = i_count_mat.shape[0]
    numerator = i_count_mat + i_count_mat.T

    non_zero_idx = (0 != numerator)
    count_mat_row_sum = np.sum(i_count_mat, axis=1)

    # compute initial value x0
    x0 = compute_init_value(i_count_mat)

    x = x0.copy()
    del x0

    it = 0
    denominator = np.zeros(x.shape, dtype=np.float64)

    while it < i_max_iter:
        x_row_sum = np.sum(x, axis=1)

        # q_i = c_i / x_i
        q_i = count_mat_row_sum / x_row_sum

        for i in xrange(n):
            for j in xrange(n):
                denominator[i, j] = q_i[i] + q_i[j]
        x[non_zero_idx] = numerator[non_zero_idx] / denominator[non_zero_idx]

        it += 1

    est_trans_mat = x / np.sum(x, axis=1)
    return est_trans_mat

