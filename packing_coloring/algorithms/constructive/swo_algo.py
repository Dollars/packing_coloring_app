# -*- coding: utf-8 -*-

import numpy as np

from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.constructive.neighborhood.random import *
from packing_coloring.algorithms.constructive.neighborhood.closeness import *
from packing_coloring.algorithms.constructive.neighborhood.betweenness import *
from packing_coloring.algorithms.constructive.greedy_algo import *
from packing_coloring.algorithms.constructive.rlf_algo import *


# Squeaky Wheel Optimizer
# TODO: be careful with pack-col where the packing chromatic number is greater than the graph's diameter.
def swo_algorithm(prob, iter_count=500, blame_value=5, blame_rate=0.75, pack_rate=0.25, random_init=True):
    if random_init:
        priority_seq = random_order(prob)
    else:
        priority_seq = (rlf_algorithm(prob)).get_greedy_order()
    best_score = float("inf")
    best_sol = None

    for i in np.arange(iter_count):
        cur_sol = greedy_algorithm(prob, priority_seq)
        cur_score = cur_sol.get_max_col()

        if cur_score < best_score:
            best_sol = cur_sol
            best_score = best_sol.get_max_col()

        b_v = max(blame_value, np.sum(cur_sol >= prob.diam))
        blame = np.arange(prob.v_size, dtype=int)
        for j, v in enumerate(priority_seq):
            if cur_sol[v] > blame_rate * min(cur_score, prob.diam):
                # blame[j] = blame[j] - np.ceil(blame_value * (1 + (cur_sol[j] / cur_score)))
                # blame[j] = (blame[j]*(i/iter_count)) - blame_value
                # blame[j] = (blame[j]*(i/iter_count)) - np.ceil(blame_value * (1 + (cur_sol[j] / cur_score)))
                # blame[j] = blame[j] - (blame_value + (cur_sol[j] * pack_rate))
                blame[j] = blame[j] - max(b_v, cur_sol.pack_size(cur_sol[v]-1))
                # blame[j] = blame[j] - b_v

        priority_seq = priority_seq[np.lexsort((blame - np.arange(prob.v_size), blame))]

    return best_sol
