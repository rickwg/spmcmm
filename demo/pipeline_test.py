#!/usr/bin/python
import clustering.edit as cl
import numpy as np
import estimation.estimation as est
import simpleDataImporter as dataImp
import analysis.analysis as ana
import argparse

parser = argparse.ArgumentParser(description='Test of spmcmm pipeline.')
parser.add_argument('-iFile', default='data/MCMM/example_1.dat', help='the input file')
parser.add_argument('-oPath', default='demo/', help='the output path')
parser.add_argument('-numCluster', default=3, type=int,  help='number of clusters of the k-means algorithm')
parser.add_argument('-plot', default=True, help='show plots')

args = parser.parse_args()

if args.plot:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print 'No matplotlib available, import failed.'

#####################################################################
# import data
#####################################################################
data_importer = dataImp.SimpleDataImporter(args.iFile, i_delimiter=' ')

#####################################################################
# k-means clustering
#####################################################################
cluster = cl.kMeans(data_importer.get_data(), args.numCluster)
cluster.discretize()

#####################################################################
# estimation of transition matrix
#####################################################################
chain = np.asarray(cluster.get_labels(), dtype=np.int64)
count_mat = est.compute_count_matrix(chain, i_tau=1)
trans_mat = est.estimate_transition_matrix(count_mat, 10000, 1e-3)

is_reversible, check_mat = est.check_reversibility(trans_mat)

#####################################################################
# Analysis of markov chain and transition matrix
#####################################################################
mcmm_ana = ana.MarkovModel(trans_mat)