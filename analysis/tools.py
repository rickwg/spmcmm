#!/usr/bin/python

import numpy as np

def is_transition_matrix(P):
	"""
	Compute the pcca of a transition matrix
	
	Parameters
	----------
	P : (nxn) ndarray
	transition matrix

	Returns
	-------
	bool : 	True if the matrix is a transition matrix
			False either
	"""
	try:
		#sum of all the lines must be one
		assert np.sum(P, axis=1) == [1]*P.shape[0]
		return True
	except:
		return False

def depth_first_search(M, node, node_list, visited_nodes=None):
	"""
	Perform depth first search algorithm

	Parameters
	----------
	M : (nxn) ndarray
	transition matrix

	node : int
	vertex of M

	node_list : (l) list
	list of visited nodes

	"""
	visited_nodes = visited_nodes or set()
	# mark the current vertex as visited
	visited_nodes.add(node)

	adjacent_nodes = set(np.where(M.T[node, :] > 0.)[0].tolist())
	adjacent_nodes.discard(node)

	for recursive_node in adjacent_nodes:
		# explore adjacent vertex that are not visited yet
		if recursive_node not in visited_nodes:
			depth_first_search(M, recursive_node, node_list, visited_nodes)

	if node not in node_list:
		node_list.append(node)

def communication_classes(P):

	"""
	Perform Kosaraju algorithm to find the strongly connected components of
	a directed graph

	Parameters
	----------
	P : (nxn) ndarray
	transition matrix

	Returns
	-------
	communication_classes : (n) list
	communication classes of the transition matrix P
	"""

	node_list = [s]
	communication_classes = []
	n_nodes = P.shape[0]

	# while node_list doesn't contain all vertices
	while(len(node_list) < n_nodes):
		# choose a vertex node not in node_list
		node = [node for node in range(n_nodes) if node not in node_list][0]
		# depth-first search starting at node
		depth_first_search(P, node, node_list)

	# transpose graph
	reverse_graph = np.copy(np.transpose(P))

	# while node_list contains nodes
	while(len(node_list) > 0):
		# remove the last vertex node
		node = node_list.pop()

		comm_class = []
		depth_first_search(reverse_graph, node, comm_class)
		communication_classes.append(comm_class)

		# remove all these vertices from the transposed graph
		for x in comm_class:
			reverse_graph[x, :] = 0.
			reverse_graph[:, x] = 0.

		node_list = [x for x in node_list if x not in comm_class]

	return communication_classes


def is_irreducible(P):
	"""
	Check if a matrix is irreducible
	"""
	return len(communication_classes(P)) == 1