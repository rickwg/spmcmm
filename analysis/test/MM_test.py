import numpy as np
import unittest
from analysis import MarkovModel

class TestAnalysis(unittest.TestCase):
	def test_eigenVectors(self):
		T = np.array([[1,0,0],[0,3,0],[0,0,4]])
		eVec_th = np.array([[1.,0.,0.], [0.,1.,0.],[0.,0.,1.]])
		eVec_an = MarkovModel(T, test=True).eigenVectors()[1]

		eVec_th = sorted(map(lambda x : list(x), eVec_th))
		eVec_an = sorted(map(lambda x : list(x), eVec_an))

		self.assertTrue(np.allclose(eVec_an,eVec_th , atol=1e-5))

	def test_eigenValues(self):
		T = np.array([[1,0,2],[0,3,1],[0,0,-4]])
		eVal_th = sorted([1.,3.,-4.])
		eVal_an = sorted(MarkovModel(T, test=True).eigenVectors()[0])
		self.assertTrue(np.allclose(eVal_an, eVal_th, atol=1e-5))

	def test_timescales(self):
		"""
		Tests if the implemented timescales method (analysis.timescales) works
		with a simple transition matrix (taken from http://www.pythonhosted.org/
		pyEMMA/api/generated/pyemma.msm.analysis.timescales.html)
		"""
		T = np.array([[1,0,0],[0,3,0],[0,0,4]])

		timescales_an = sorted(MarkovModel(T, test=True).timescales())
		timescales_th = sorted(np.array([np.inf, -1./np.log(3), -1./np.log(4)]))
		
		self.assertTrue(np.allclose(timescales_an, timescales_th, rtol=1.e-5))

if __name__ == '__main__':
	unittest.main()