# -*- coding: utf-8 -*-

import graph_tool.all as gt
import numpy as np
from packing_coloring.utils.graph_utils import *


def betweenness_order(prob):
    g = graph_from_adj(prob.adj_matrix)

    v_bet, e_bet = gt.betweenness(g)
    bet_index = np.argsort(v_bet.a)
    return bet_index
