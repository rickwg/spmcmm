import numpy as np


def compute_count_matrix(i_chain):
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

    for i in xrange(1, i_chain.shape[0]):
        count_matrix[i_chain[i - 1], i_chain[i]] += 1
    return count_matrix


def estimate_transition_matrix_naiv(i_count_mat):
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

def estimate_transition_matrix(i_count_mat, i_x0, i_max_iter, i_tol):
    """
    Estimate transition matrix with detailed balance condition

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
    count_mat_row_sum = np.sum(i_count_mat, axis=1)
    count_mat_col_sum = np.sum(i_count_mat, axis=0)
    x = i_x0
    iterations = 0
    denominator = np.zeros(x.shape)

    while iter < i_max_iter:
        x_row_sum = np.sum(x, axis=1)
        # x_col_sum = np.sum(x, axis=0)

        # q_i = c_i / x_i
        q_i = count_mat_row_sum / x_row_sum
        q_j = count_mat_col_sum / x_row_sum

        for i in xrange(n):
            for j in xrange(n):
                denominator[i, j] = q_i[i] + q_j[j]
        x = numerator / denominator

        iterations += 1

    est_trans_mat = x / np.sum(x, axis=1)
    return est_trans_mat

