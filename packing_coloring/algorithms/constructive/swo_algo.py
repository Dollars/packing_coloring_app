# -*- coding: utf-8 -*-

import numpy as np

from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.constructive.neighborhood.random import *
from packing_coloring.algorithms.constructive.neighborhood.closeness import *
from packing_coloring.algorithms.constructive.neighborhood.betweenness import *
from packing_coloring.algorithms.constructive.greedy_algo import *


# Squeaky Wheel Optimizer
def swo_algorithm(prob, iter_count=500, blame_value=5, blame_rate=0.75):
    priority_seq = random_order(prob)
    best_score = float("inf")
    best_sol = None

    for i in np.arange(iter_count):
        cur_sol = greedy_algorithm(prob, priority_seq)
        cur_score = cur_sol.get_max_col()
        if cur_score < best_score:
            best_sol = cur_sol
            best_score = best_sol.get_max_col()
            print(best_score)

        blame = np.arange(prob.v_size)
        for j in np.arange(prob.v_size):
            if cur_sol[j] > (blame_rate * cur_score):
                blame[j] = blame[j] - blame_value

        print("blame: ", blame)
        print("current pri seq: ", priority_seq)
        priority_seq = priority_seq[np.argsort(blame)]
        print("new pri seq: ", priority_seq, "")

    return best_sol
