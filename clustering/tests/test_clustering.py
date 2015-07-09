import unittest
import numpy as np
from clustering.edit import kMeans


class TestKmeans(unittest.TestCase):
    def test_kmeans(self):

            testOneD = np.array((1, 1.1, 0.9, 130, 130.1, 129.9))
            testOneD = testOneD.reshape(testOneD.size,1)
            k = 2

            cluster = kMeans(testOneD, k)
            cluster.discretize()
            self.assertEqual(sorted(cluster.get_centers().tolist()), [[1.], [130.]])
            self.assertEqual(sorted(cluster.get_labels().tolist()), [[0], [0], [0], [1], [1], [1]])

            testTwoD = np.array(((1,0),(1,2),(2,1),(15,4),(13,2),(21,4)))
            centerY = np.array(((0,0),(20,5)))

            cluster2 = kMeans(testTwoD)
            cluster2.set_centers(centerY)
            cluster2.discretize()
            self.assertEqual(sorted(cluster2.get_centers().tolist()), [[0,0], [20,5]])
            self.assertEqual(sorted(cluster2.get_labels().tolist()), [[0], [0], [0], [1], [1], [1]])


if __name__ == '__main__':
    unittest.main()