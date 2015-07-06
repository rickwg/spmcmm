import numpy as np

from nose.tools import assert_true

import estimation.estimation as est


def test_compute_count_mat():
    """Testing computation of count matrix"""
    check_mat = np.array([[0, 7, 0], [0, 0, 6], [6, 0, 0]])
    chain = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    count_mat = est.compute_count_matrix(chain)

    assert_true(count_mat == check_mat)

def test_est_trans_matrix_naive():
    """Testing estimation of transition matrix"""
    trans_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    count_mat = np.array([[0, 7, 0], [0, 0, 6], [6, 0, 0]])
    est_trans_mat = est.estimate_transition_matrix_naive(count_mat)

    assert_true(est_trans_mat == trans_mat)
