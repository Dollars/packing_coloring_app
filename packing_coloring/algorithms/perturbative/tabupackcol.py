# -*- coding: utf-8 -*-

import numpy as np
import numpy
from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm

class TabuPackCol:
    def __init__(self):
        pass

    def tabu_search(self, dist_mat, pack_coloring, k_col, max_iter=1000):
        tabu_list = np.zeros((len(pack_coloring), k_col), dtype=int)
        pack_coloring[pack_coloring >= k_col] = 0

    def solve(self, dist_mat, colors=0):
        k_colors = colors
