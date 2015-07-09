#!/usr/bin/python

import numpy as np

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

    adjacentNodes = set(np.where(M.T[node, :] > 0.)[0].tolist())
    adjacentNodes.discard(node)

    for recursNode in adjacentNodes:
        # explore adjacent vertex that are not visited yet
        if recursNode not in visited_nodes:
            depth_first_search(M, recursNode, node_list, visited_nodes)

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

    node_list = []
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
