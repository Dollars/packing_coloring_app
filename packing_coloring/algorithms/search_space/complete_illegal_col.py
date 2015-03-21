# -*- coding: utf-8 -*-

import numpy as np

def conflicting_vertices(prob, sol):
    conflicted = np.zeros(prob.v_size, dtype=bool)
    for v in range(prob.v_size):
        v_col = sol[v]
        v_conflict = np.logical_and(sol.pack_col == v_col, prob.dist_matrix[v] <= v_col)
        v_conflict[v] = False
        conflicted = np.logical_or(conflicted, v_conflict)

    return conflicted

