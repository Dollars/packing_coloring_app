# -*- coding: utf-8 -*-

import numpy as np
from packing_coloring.utils.graph_utils import *


def random_order(prob):
    g = graph_from_adj(prob.adj_matrix)

    rand_index = np.arange(g.num_vertices())
    np.random.shuffle(rand_index)
    return rand_index
