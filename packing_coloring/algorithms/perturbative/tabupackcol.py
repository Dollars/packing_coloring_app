# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd
import time

from packing_coloring.algorithms.search_space.complete_illegal_col import *
from packing_coloring.algorithms.search_space.partial_valide_col import *
from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm


def tabu_kpack_col(prob, k_col, sol=None, tt_a=10, tt_d=0.5, max_iter=1000):
    colors = np.arange(1, k_col+1)
    tabu_list = np.zeros((prob.v_size, k_col), dtype=int)

    if sol is None:
        sol = rlf_algorithm(prob)
    if k_col < prob.get_diam():
        sol[sol > k_col] = rd.randint(1, k_col+1, len(sol[sol > k_col]))
    else:
        sol[sol > prob.get_diam()] = rd.randint(
            1, k_col+1, len(sol[sol > prob.get_diam()]))

    score = count_conflict(prob, sol)
    while score > 0 and max_iter > 0:
        vertex, col = best_one_exchange(prob, sol, colors, tabu_list)

        prev_col = sol[vertex]
        sol[vertex] = col

        for v in np.arange(prob.v_size)[sol == 0]:
            v_col_tt = colors[tabu_list[v] == 1]
            if len(v_col_tt) > 0:
                sol[v] = v_col_tt[0]
        score = count_conflict(prob, sol)

        tabu_list = tabu_list - 1
        tabu_list[tabu_list < 0] = 0
        tabu_list[vertex, prev_col-1] = (
            rd.randint(tt_a) + (tt_d * score * prev_col))

        max_iter -= 1
    return sol


def tabu_pack_col(prob, k_count=3, sol=None, tt_a=10, tt_d=0.5, max_iter=1000, duration=30):
    end_time = time.time()+(duration*60)

    if sol is None:
        sol = rlf_algorithm(prob)
    lim_col = sol.get_max_col()
    best_sol = sol.copy()
    new_sol = sol.copy()

    count = 0
    k_col = lim_col - 1
    k_lim = lim_col
    while k_col < lim_col and count < k_count:
        print(k_col, flush=True)
        new_sol = tabu_kpack_col(prob, k_col, new_sol, tt_a, tt_d, max_iter)
        new_score = count_conflict(prob, new_sol)
        if new_score == 0:
            k_lim = k_col
            k_col = k_col - 1
            if new_sol.get_max_col() < best_sol.get_max_col():
                count = 0
                best_sol = new_sol.copy()
        else:
            k_col = k_col + 1
            if k_col >= k_lim:
                count += 1

        if time.time() >= end_time:
            break

    return best_sol


def partial_kpack_col(prob, k_col, sol=None, tt_a=50, tt_d=0.5, max_iter=1000):
    colors = np.arange(1, k_col+1)
    tabu_list = np.zeros((prob.v_size, k_col), dtype=int)

    if sol is None:
        sol = rlf_algorithm(prob)
    if k_col < prob.get_diam():
        sol[sol > k_col] = 0
    else:
        sol[sol > prob.get_diam()] = 0

    score = float("inf")
    while score > 0 and max_iter > 0:
        vertex, col = best_i_swap(prob, sol, colors, tabu_list)
        # print(vertex, col, ": ", score)

        prev_colored = (sol == col)
        sol = assign_col(prob, sol, col, vertex)
        score = sol.count_uncolored()

        tabu_list = tabu_list - 1
        tabu_list[tabu_list < 0] = 0

        decolored = np.logical_and(prev_colored, sol == 0)
        for v in np.arange(prob.v_size)[decolored]:
            tabu_list[v, col-1] = (
                rd.randint(tt_a) + (tt_d * score * col))

        max_iter -= 1

    if np.any(sol == 0):
        sol = rlf_algorithm(prob, sol)

    return sol


def partial_pack_col(prob, k_count=3, sol=None, tt_a=10, tt_d=0.5, max_iter=1000, duration=30):
    end_time = time.time()+(duration*60)

    if sol is None:
        sol = rlf_algorithm(prob)
    lim_col = sol.get_max_col()
    best_sol = sol.copy()
    new_sol = sol.copy()

    count = 0
    k_col = lim_col - 1
    k_lim = lim_col
    while count < k_count:
        print(k_col)
        new_sol = partial_kpack_col(prob, k_col, new_sol, tt_a, tt_d, max_iter)
        if new_sol.get_max_col() == k_col:
            k_lim = k_col
            k_col = k_col - 1
            if new_sol.get_max_col() < best_sol.get_max_col():
                count = 0
                best_sol = new_sol.copy()
        else:
            k_col = k_col + 1
            if k_col >= k_lim:
                count += 1

        if time.time() >= end_time:
            break

    return best_sol

