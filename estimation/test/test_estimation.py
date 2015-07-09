import numpy as np
import unittest

from estimation import estimation as est

class TestEstimation(unittest.TestCase):
    def test_compute_count_mat(self):
        """Testing computation of count matrix"""
        check_mat = np.array([[0, 7, 0], [0, 0, 6], [6, 0, 0]])
        chain = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        count_mat = est.compute_count_matrix(chain)

        self.assertTrue(np.allclose(count_mat, check_mat, atol=1e-5))

    def test_est_trans_matrix_naive(self):
        """Testing estimation of transition matrix"""
        trans_mat = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        count_mat = np.array([[0, 7, 0], [0, 0, 6], [6, 0, 0]])
        est_trans_mat = est.estimate_transition_matrix_naive(count_mat)

        self.assertTrue(np.allclose(est_trans_mat, trans_mat, atol=1e-5))

    def test_compute_count_matrix_list(self):
        """Test computation of count matrix for a list of trajectories"""
        check_mat = np.array([[0, 14, 0], [0, 0, 12], [12, 0, 0]])
        chain = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        count_mat = est.compute_count_matrix_list([chain, chain])

        self.assertTrue(np.allclose(count_mat, check_mat, atol=1e-5))

    def test_compute_init_value(self):
        """
        Test computation of initial value of the function:
        estimate_transition_matrix.
        """
        count_mat = np.array([[1, 3], [3, 1]])
        check_mat = np.array([[0.1767767,  0.53033009], [0.53033009,  0.1767767]])
        init_value = est.compute_init_value(count_mat)

        self.assertTrue(np.allclose(init_value, check_mat, atol=1e-5))

    def test_estimate_transition_matrix(self):
        """
        Test estimation of reversible transition matrix
        """
        check_mat = np.array([[0.25, 0.75], [0.75, 0.25]])
        count_mat = np.array([[1, 3], [3, 1]])
        trans_mat = est.estimate_transition_matrix(count_mat, i_max_iter=10000, i_tol=1e-6)

        self.assertTrue(np.allclose(trans_mat, check_mat, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
