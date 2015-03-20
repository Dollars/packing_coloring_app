# -*- coding: utf-8 -*-

import numpy as np
import graph_tool.all as gt
from packing_coloring.utils.graph_utils import *

class GraphProblem:
    def __init__(self, graph):
        # Define the distance matrix,
        # which is used to know the topologie of the problem's graph 
        self.dist_matrix = np.matrix([], dtype=int)
        if type(graph) is gt.Graph:
            dist = gt.shortest_distance(graph)
            dim = g.num_vertices()
            for v in g.vertices():
                self.dist_matrix = np.append(self.dist_matrix, 
                    np.array(dist[v], ndmin=2), axis=0)
            self.dist_matrix = np.delete(self.dist_matrix, 0, 0)
            self.dist_matrix = np.asmatrix(self.dist_matrix)
        else:
            d_mat = np.matrix([], dtype=int)
            if type(graph) is np.ndarray:
                d_mat = np.asmatrix(graph)

            elif type(graph) is np.matrix:
                d_mat = np.copy(graph)

            elif type(graph) is list:
                d_mat = np.matrix(graph)

            else:
                #TODO: throw exception
                pass

            if np.all(np.logical_or(d_mat == 0, d_mat == 1)):
                self.dist_matrix = graph_from_adj(d_mat)
            else:
                self.dist_matrix = d_mat

        if self.dist_matrix.shape[0] != self.dist_matrix.shape[1]:
            #TODO: throw exception
            pass

        self.v_size = self.dist_matrix.shape[0]
        self._closeness_values = None
        self._betweenness_values = None

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

