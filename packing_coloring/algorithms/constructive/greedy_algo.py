# -*- coding: utf-8 -*-

import numpy as np

class greedy_algorithm:

    def __init__(self, heur):
        self.heuristic = heur

    def assign_min_pcol(self, dist_mat, coloring, vi):
        vi_dist = dist_mat[vi]
        for k_col in np.arange(1, len(coloring)):
            is_k_col = (coloring == k_col)
            is_k_dist = (vi_dist <= k_col)
            is_k_dist[vi] = False
            is_conflict = np.logical_and(is_k_col, is_k_dist)

            if not is_conflict.any():
                coloring[vi] = k_col
                break

        return coloring

    def solve(self, dist_mat):
        coloring = np.zeros(dist_mat.shape[0])
        vertex_order = self.heuristic(dist_mat == 1)
        coloring[vertex_order[0]] = 1

        for v in vertex_order[1:]:
            self.assign_min_pcol(dist_mat, coloring, v)

        return coloring
