# -*- coding: utf-8 -*-

import numpy as np

from packing_coloring.algorithms.search_space.complete_illegal_col import *
from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm

def tabu_kpack_col(prob, sol, k_col, max_iter=0):
    tabu_list = np.zeros((prob.v_size, k_col), dtype=int)
    

def tabu_pack_col(prob, col, max_iter=0):
