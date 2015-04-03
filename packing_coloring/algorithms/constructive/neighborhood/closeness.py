# -*- coding: utf-8 -*-

import graph_tool.all as gt
import numpy as np
from packing_coloring.utils.graph_utils import *


def closeness_order(prob):
    g = graph_from_adj(prob.adj_matrix)

    v_clos = gt.closeness(g)
    clos_index = np.argsort(v_clos.a)
    return clos_index
