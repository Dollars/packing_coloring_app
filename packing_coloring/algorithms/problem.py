# -*- coding: utf-8 -*-
import numpy as np
import graph_tool.all as gt
from packing_coloring.utils.graph_utils import get_distance_matrix
from packing_coloring.utils.graph_utils import graph_from_adj, graph_from_dist


class GraphProblem:
    def __init__(self, graph):
        # Define the distance matrix,
        # which is used to know the topologie of the problem's graph
        self.dist_matrix = np.matrix([], dtype=int)
        if type(graph) is gt.Graph:
            self.dist_matrix = get_distance_matrix(graph)
        else:
            d_mat = np.matrix([], dtype=int)
            if type(graph) is np.ndarray:
                d_mat = np.matrix(graph)

            elif type(graph) is np.matrix:
                d_mat = np.matrix(np.copy(graph))

            elif type(graph) is list:
                d_mat = np.matrix(graph)

            else:
                # TODO: throw exception
                pass

            if np.all(np.logical_or(d_mat == 0, d_mat == 1)):
                g = graph_from_adj(d_mat)
                self.dist_matrix = get_distance_matrix(g)
            else:
                self.dist_matrix = d_mat

        if self.dist_matrix.shape[0] != self.dist_matrix.shape[1]:
            # TODO: throw exception
            pass

        self.v_size = self.dist_matrix.shape[0]
        self.diam = np.max(self.dist_matrix)
        self._closeness_values = None
        self._betweenness_values = None
        self._avg_kdegree = None
        self.name = "{0}-{1}".format(self.v_size, int(np.sum(self.dist_matrix == 1) / 2))

    def get_diam(self):
        return self.diam

    def avg_kdegree(self, k_col):
        if self._avg_kdegree is None:
            self._avg_kdegree = np.zeros(self.diam, dtype=float)
            self._avg_kdegree[:] = -1.

        if k_col < self.diam:
            if self._avg_kdegree[k_col] == -1.:
                norm = 1./self.v_size
                kdist_degree = self.dist_matrix <= k_col
                self._avg_kdegree[k_col] = norm * np.sum(kdist_degree)
            return self._avg_kdegree[k_col]
        elif k_col >= self.diam:
            return 1
        else:
            return 0

    def __getitem__(self, key):
        try:
            return self.dist_matrix[key].A1
        except IndexError:
            raise IndexError("index out of bound: {0}".format(key))

    @property
    def adj_matrix(self):
        return self.dist_matrix == 1

    @property
    def closeness_values(self):
        if self._closeness_values is None:
            g = graph_from_dist(self.dist_matrix)
            v_clos = gt.closeness(g)
            self._closeness_values = v_clos.a
        return self._closeness_values

    @property
    def betweenness_values(self):
        if self._betweenness_values is None:
            g = graph_from_dist(self.dist_matrix)
            v_bet, e_bet = gt.betweenness(g)
            self._betweenness_values = v_bet.a
        return self._betweenness_values
