
from nose.tools import assert_true
import estimation.estimation as est
import estimation.naive_sampling as nsampl


import numpy as np

def test_compute_count_mat():
    """Testing computation of count matrix"""
    check_mat = np.array([[0, 3, 0], [0, 0, 3], [3, 0, 0]], dtype=np.intc)

    trans_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.intc)
    chain = nsampl.evolve_chain(0, trans_mat, 10)
    count_mat = est.compute_count_matrix(chain)

    assert_true(count_mat == check_mat)

def test_est_trans_matrix_naive():
    """Testing estimation of transition matrix"""
    trans_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.intc)
    chain = nsampl.evolve_chain(0, trans_mat, 10)
    count_mat = est.compute_count_matrix(chain)
    est_trans_mat = est.estimate_transition_matrix_naive(count_mat)

    assert_true(est_trans_mat == trans_mat)
