import numpy as np
import unittest
from analysis.analysis import TPT

class TestTPT(unittest.TestCase):
    def test_forwardCommittor(self):
        '''
        Test the computation of the forward committor
        '''
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        forwardCommit_an = TPT(T,[0],[2]).forwardCommittor()
        forwardCommit_th = np.array([0.,0.8,1.]).T
        self.assertTrue(np.allclose(forwardCommit_an,forwardCommit_th, atol=1e-5))
    def test_backwardCommittor(self):
        '''
        Test the computation of the backward committor
        '''     
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        backwardCommit_an = TPT(T,[0],[2]).backwardCommittor()
        backwardCommit_th = np.array([1.,0.2,0.]).T
        self.assertTrue(np.allclose(backwardCommit_an,backwardCommit_th, atol=1e-5))

    def test_probabilityCurrent(self):
        '''
        Test the computation of the probability current
        '''
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        probCurrent_an = TPT(T,[0],[2]).probabilityCurrent()
        probCurrent_th = np.array([[0.,0.09889961,0.],[0.,0.,-0.02569381],[0.,0.,0.]])
        self.assertTrue(np.allclose(probCurrent_th,probCurrent_an,atol=1e-5))

    def test_effectiveProbabilityCurrent(self):
        '''
        Test the computation of the effective probability current
        '''
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        effProbCurrent_th = np.array([[0.,0.09889961,0.],[0.,0.,0.],[0.,0.02569381,0.]])
        effProbCurrent_an = TPT(T,[0],[2]).effectiveProbabilityCurrent()
        self.assertTrue(np.allclose(effProbCurrent_th,effProbCurrent_an,atol=1e-5))
    
    def test_flux(self):
        '''
        Test the computation of the flux
        '''
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        flux_th = 0.09889961
        flux_an = TPT(T,[0],[2]).filux()
        self.assertTrue(np.allclose(flux_an,flux_th,atol=1e-5))
    
    def test_transitionRate(self):
        '''
        Test the computation of the transition rate
        '''
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        transitionRate_an = TPT(T,[0],[2]).transitionrate()
        transitionRate_th =  0.09889961/0.10524330
        self.assertTrue(np.allclose(transitionRate_th,transitionRate_an,atol=1e-5))
    
    def test_meanFirstPassageTime(self):
        '''
        Test the computation of the mean first passage time
        '''
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        meanFirstPassageTime_an = TPT(T,[0],[2]).meanfirstpassagetime()
        meanFirstPassageTime_th = 0.10524330/0.09889961
        self.assertTrue(np.allclose(meanFirstPassageTime_an,meanFirstPassageTime_th,atol=1e-5))
    
    def test_minCurrent(self):
        '''
        Test the computation of the min current of a pathway
        '''
        T = np.array([[0.1,0.9,0],[0.2,0,0.8],[0.,0.3,0.7]])
        meanCurrent_an = TPT(T,[0],[2]).minCurrent([0,1,2])
        meanCurrent_th = 0.
        self.assertTrue(np.allclose(meanCurrent_an,meanCurrent_th,atol=1e-5))

if __name__ == '__main__':
    unittest.main()