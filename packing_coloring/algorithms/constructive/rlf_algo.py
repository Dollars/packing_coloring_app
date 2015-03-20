# -*- coding: utf-8 -*-

import numpy as np
from packing_coloring.algorithms.search_space.partial_valide_col import *
from packing_coloring.algorithms.solution import *


def rlf_algorithm(prob):
    vert_num = prob.v_size
    coloring = PackColSolution(prob)

    # The coloring is done by constructing the k-packing
    k_col = 1
    while not coloring.is_complete():
        v = partition_next_vertex(prob, coloring, k_col)
        coloring[v] = k_col

        while np.any(k_colorable_set(prob, coloring, k_col)):
            v = partition_next_vertex(prob, coloring, k_col)
            coloring[v] = k_col

        k_col += 1

    return coloring

