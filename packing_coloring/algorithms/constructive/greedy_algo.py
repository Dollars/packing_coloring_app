# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from packing_coloring.algorithms.solution import PackColSolution
from packing_coloring.utils.benchmark_utils import search_step_trace


@search_step_trace
def assign_min_pcol(prob, coloring, vi):
    vi_dist = prob[vi]

    k_col = 1
    while k_col < len(coloring):
        is_k_col = (coloring == k_col)
        is_k_dist = (vi_dist <= k_col)
        is_k_dist[vi] = False
        is_conflict = np.logical_and(is_k_col, is_k_dist)

        if not is_conflict.any():
            coloring[vi] = k_col
            break

        k_col = k_col + 1

    return coloring


@search_step_trace
def greedy_algorithm(prob, ordering):
    coloring = PackColSolution(prob)
    if callable(ordering):
        vertex_order = ordering(prob)
    else:
        vertex_order = ordering

    coloring[vertex_order[0]] = 1

    for v in vertex_order[1:]:
        assign_min_pcol(prob, coloring, v)

    coloring.get_trace()
    return coloring
