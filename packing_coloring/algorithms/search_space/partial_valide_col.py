# -*- coding: utf-8 -*-

import numpy as np


def k_colorable_set(prob, sol, k_col):
    k_colorable = sol.uncolored()
    k_colored = (sol.pack_col == k_col)
    for v in np.arange(prob.v_size)[k_colored]:
        good_dist = (prob.dist_matrix[v].A1 > k_col)
        k_colorable = np.logical_and(k_colorable, good_dist)
    return k_colorable


def k_uncolorable_set(prob, sol, k_col):
    k_uncolorable = np.zeros_like(sol.colored())
    k_colored = (sol.pack_col == k_col)
    for v in np.arange(prob.v_size)[k_colored]:
        bad_dist = (prob.dist_matrix[v].A1 <= k_col)
        k_uncolorable = np.logical_or(k_uncolorable, bad_dist)
    k_uncolorable = np.logical_and(k_uncolorable, sol.uncolored())
    
    return k_uncolorable


def coloried_and_k_uncolorable_set(prob, sol, k_col):
    k_uncolorable = sol.colored()
    k_colored = (sol.pack_col == k_col)
    for v in np.arange(prob.v_size)[k_colored]:
        bad_dist = (prob.dist_matrix[v].A1 <= k_col)
        k_uncolorable = np.logical_or(k_uncolorable, bad_dist)
    return k_uncolorable


# TODO: add sumplementary sorting criterion ?
def partition_next_vertex(prob, sol, k_col):
    vertices = np.arange(prob.v_size)
    k_col_set = k_colorable_set(prob, sol, k_col)
    k_uncol_set = k_uncolorable_set(prob, sol, k_col)
    col_and_k_uncol_set = coloried_and_k_uncolorable_set(prob, sol, k_col)

    k_uncol_dist_mat = prob.dist_matrix[k_uncol_set]
    gt_unk_dist_score = np.sum(k_uncol_dist_mat > k_col, axis=0).A1

    # second ordering: the score is the number of uncolored
    # vertices which will not be k-colorable anymore
    k_col_dist_mat = prob.dist_matrix[k_col_set]
    k_dist_score = np.sum((k_col_dist_mat <= k_col), axis=0).A1

    # third ordering: the score is the number of vertices
    # which are colored or not k-colorable
    # with a distance greater than k
    col_and_k_uncol_dist_mat = prob.dist_matrix[col_and_k_uncol_set]
    gt_k_dist_score = np.sum(col_and_k_uncol_dist_mat <= k_col, axis=0).A1
    gt_k_dist_score = prob.v_size - gt_k_dist_score

    # Works better on a grid, this is a kind of DSATUR for k dist
    influence = np.lexsort((gt_unk_dist_score, gt_k_dist_score, k_dist_score))

    v = influence[np.in1d(influence, vertices[k_col_set])][0]
    return v


def conflicting_vertices(prob, sol):
    conflicting = np.zeros(prob.v_size, dtype=bool)
    for v in range(prob.v_size):
        v_col = sol[v]
        v_conflict = np.logical_and(
            sol.pack_col == v_col, prob[v] <= v_col)
        v_conflict[v] = False
        conflicting = np.logical_or(conflicting, v_conflict)
    return conflicting_vertices


def assign_col(prob, sol, k_col, v):
    new_sol = sol.copy()
    v_conflict = np.logical_and(
            new_sol.pack_col == k_col, prob[v] <= k_col)
    new_sol.pack_col[v_conflict] = 0
    new_sol[v] = k_col
    return new_sol


def best_i_swap(prob, sol, colors, tabu_list=None):
    vertices = np.arange(prob.v_size)
    curent_score = sol.count_uncolored()
    best_score = float("inf")
    best_sum = float("inf")
    changed_v = -1
    changed_col = 0

    for v in vertices[sol.uncolored()]:
        for col in colors:
            new_sol = assign_col(prob, sol, col, v)
            new_score = new_sol.count_uncolored()
            new_sum = new_sol.get_sum()

            if new_score < best_score or (new_score == best_score and new_sum < best_sum):
                if tabu_list[v, col-1] == 0:
                        best_score = new_score
                        best_sum = new_sum
                        changed_v = v
                        changed_col = col
                else:
                    if new_score < curent_score:
                        best_score = new_score
                        best_sum = new_sum
                        changed_v = v
                        changed_col = col

    return changed_v, changed_col
