# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd


def conflicting_vertices(prob, sol):
    conflicting = np.zeros(prob.v_size, dtype=bool)
    for v in range(prob.v_size):
        v_col = sol[v]
        v_conflict = np.logical_and(
            sol.pack_col == v_col, prob[v] <= v_col)
        v_conflict[v] = False
        conflicting = np.logical_or(conflicting, v_conflict)
    return conflicting


def count_conflicting_vertex(prob, sol):
    score = np.sum(conflicting_vertices(prob, sol))
    sol.score = score
    return score


def count_conflict(prob, sol):
    conflicting = 0
    for v in range(prob.v_size):
        v_col = sol[v]
        v_conflict = np.logical_and(
            sol.pack_col == v_col, prob[v] <= v_col)
        v_conflict[v] = False
        conflicting = conflicting + np.sum(v_conflict)
    sol.score = conflicting
    return conflicting


def best_one_move(prob, sol, colors, tabu_list=None):
    vertices = np.arange(prob.v_size)
    curent_score = count_conflict(prob, sol)
    best_score = float("inf")
    changed_v = -1
    changed_col = 0

    for v in vertices[conflicting_vertices(prob, sol)]:
        v_col = sol[v]
        for col in colors:
            if col == v_col:
                continue
            new_sol = sol.copy()
            new_sol[v] = col
            new_score = count_conflict(prob, new_sol)
            if tabu_list[v, col-1] > 0:
                if new_score < best_score and new_score < curent_score:
                    best_score = new_score
                    changed_v = v
                    changed_col = col
            else:
                if new_score < best_score:
                    best_score = new_score
                    changed_v = v
                    changed_col = col

    return changed_v, changed_col
