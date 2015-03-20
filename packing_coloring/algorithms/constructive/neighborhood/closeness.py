# -*- coding: utf-8 -*-

import graph_tool.all as gt
import numpy as np
from packing_coloring.utils.graph_utils import *

def order_by_closeness(adj_mat):
    g = graph_from_adj(adj_mat)

    v_clos = gt.closeness(g)
    clos_index = np.argsort(v_clos.a)
    return clos_index