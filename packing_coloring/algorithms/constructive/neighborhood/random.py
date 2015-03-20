# -*- coding: utf-8 -*-

import numpy as np
from packing_coloring.utils.graph_utils import *

def random_order(adj_mat):
    g = graph_from_adj(adj_mat)

    rand_index = np.arange(g.num_vertices())
    np.random.shuffle(rand_index)
    return rand_index