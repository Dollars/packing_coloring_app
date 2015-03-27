# -*- coding: utf-8 -*-

import numpy as np
from packing_coloring.algorithms.solution import *


class greedy_algorithm:

    def __init__(self, heur):
        self.heuristic = heur

    def assign_min_pcol(self, prob, coloring, vi):
        vi_dist = prob[vi]
        for k_col in np.arange(1, len(coloring)):
            is_k_col = (coloring == k_col)
            is_k_dist = (vi_dist <= k_col)
            is_k_dist[vi] = False
            is_conflict = np.logical_and(is_k_col, is_k_dist)

            if not is_conflict.any():
                coloring[vi] = k_col
                break

        return coloring

    def solve(self, prob):
        coloring = PackColSolution(prob)
        vertex_order = self.heuristic(prob.dist_matrix == 1)
        coloring[vertex_order[0]] = 1

        for v in vertex_order[1:]:
            self.assign_min_pcol(prob, coloring, v)

        return coloring
