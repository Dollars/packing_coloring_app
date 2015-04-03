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
    k_uncolorable = sol.colored()
    k_colored = (sol.pack_col == k_col)
    for v in np.arange(prob.v_size)[k_colored]:
        bad_dist = (prob.dist_matrix[v].A1 <= k_col)
        k_uncolorable = np.logical_or(k_uncolorable, bad_dist)
    return k_uncolorable


# TODO: add sumplementary sorting criterion ?
def partition_next_vertex(prob, sol, k_col):
    vertices = np.arange(prob.v_size)

    # first ordering: the score is the number of uncolored
    # vertices which will not be k-colorable anymore
    k_col_set = k_colorable_set(prob, sol, k_col)
    k_col_dist_mat = prob.dist_matrix[k_col_set]
    k_dist_score = np.sum((k_col_dist_mat <= k_col), axis=0).A1

    # second ordering: the score is the number of vertices
    # which are colored or not k-colorable
    # with a distance greater than k
    k_uncol_set = k_uncolorable_set(prob, sol, k_col)
    k_uncol_dist_mat = prob.dist_matrix[k_uncol_set]
    k_uncol_dist_mat[k_uncol_dist_mat > k_col] = 0
    gt_k_dist_score = np.sum(k_uncol_dist_mat != 0, axis=0).A1
    gt_k_dist_score = prob.v_size - gt_k_dist_score

    # Works better on a grid, this is a kind of DSATUR for k dist
    # influence = np.lexsort((k_dist_score, gt_k_dist_score))

    influence = np.lexsort((gt_k_dist_score, k_dist_score))
    v = influence[np.in1d(influence, vertices[k_col_set])][0]
    return v
