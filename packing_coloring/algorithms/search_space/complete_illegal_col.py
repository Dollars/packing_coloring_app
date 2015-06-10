# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import numpy.random as rd
from packing_coloring.utils.benchmark_utils import trace


@trace
def conflicting_vertices(prob, sol):
    conflicting = np.zeros(prob.v_size, dtype=bool)
    for v in range(prob.v_size):
        v_col = sol[v]
        v_conflicts = np.logical_and(
            sol.pack_col == v_col, prob[v] <= v_col)
        v_conflicts[v] = False
        conflicting = np.logical_or(conflicting, v_conflicts)
    return conflicting


@trace
def count_conflicting_vertex(prob, sol):
    score = np.sum(conflicting_vertices(prob, sol))
    return score


@trace
def count_conflicting_edge(prob, sol):
    conflicting = 0
    for v in range(prob.v_size):
        v_col = sol[v]
        v_conflict = np.logical_and(
            sol.pack_col == v_col, prob[v] <= v_col)
        v_conflict[v] = False
        conflicting = conflicting + np.sum(v_conflict)
    return np.floor(conflicting/2)


@trace
def one_exchange(prob, sol):
    pass


@trace
def best_one_exchange(prob, sol, fitness, score,
                      the_best_score, colors, tabu_list):
    vertices = np.arange(prob.v_size)
    cur_sum = sol.get_sum()
    best_sum = float("inf")
    best_score = float("inf")
    changed_v = -1
    changed_col = 0

    for v in vertices:
        v_col = sol[v]
        for col in colors:
            if col == v_col:
                continue
            new_score = score + fitness[v, col-1]
            new_sum = cur_sum - v_col + col

            # Only used for benchmark as a counter
            one_exchange(prob, sol)

            if new_score < best_score or (
                    new_score == best_score and new_sum < best_sum):
                if tabu_list[v, col-1] == 0:
                    best_score = new_score
                    best_sum = new_sum
                    changed_v = v
                    changed_col = col
                elif new_score < the_best_score:
                    best_score = new_score
                    best_sum = new_sum
                    changed_v = v
                    changed_col = col

    return changed_v, changed_col
