# -*- coding: utf-8 -*-

import graph_tool.all as gt
import numpy as np
from packing_coloring.utils.graph_utils import *

def order_by_betweenness(adj_mat):
    g = graph_from_adj(adj_mat)

    v_bet, e_bet = gt.betweenness(g)
    bet_index = np.argsort(v_bet.a)
    return bet_index