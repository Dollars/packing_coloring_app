from __future__ import print_function

import numpy as np
import numpy.random as rd
import time

from packing_coloring.algorithms.search_space.complete_illegal_col import *
from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.problem import *
from packing_coloring.algorithms.perturbative.tabupackcol import *
from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm
from packing_coloring.algorithms.constructive.greedy_algo import greedy_algorithm


def generate_population(prob, size, heuristic, init_args):
    print("Init population by permut")
    pop = []
    indiv = heuristic(prob, random_init=False, **init_args)
    permut = np.arange(1, indiv.get_max_col()+1, dtype=int)
    pop.append(indiv)
    # print("init candidate #", 0, ":", indiv.get_max_col(), flush=True)
    for i in range(1, size):
        new_permut = rd.permutation(permut)
        priority = indiv.get_by_permut(new_permut)
        pop.append(greedy_algorithm(prob, priority))
        # print("init candidate #", i, ":", pop[i].get_max_col(), flush=True)
    return pop


def generate_population2(prob, size, heuristic, init_args):
    print("Init population by random order")
    pop = []
    for i in range(size):
        # print("init candidate #", i, flush=True)
        indiv = heuristic(prob, random_init=(i != 0), **init_args)
        pop.append(indiv)
    return pop


def generate_population3(prob, size, heuristic, init_args):
    print("Init population by RLF")
    pop = []
    for i in range(size):
        indiv = heuristic(prob, random_init=False, **init_args)
        pop.append(indiv)
    return pop


def selection(pop, pool_size=2):
    # Tournament Selection
    print("Tournament ", end="")
    indices = np.arange(len(pop), dtype=int)
    best = rd.choice(indices, 1)[0]

    for i in range(pool_size):
        adv = rd.choice(np.delete(indices, best), 1)[0]
        if pop[adv] < pop[best]:
            best = adv
    print(best)
    return pop[best]


def choose_parents(pop, nbr, pool_size):
    print("Parents selection")
    pop_indices = np.arange(len(pop), dtype=int)
    parents_i = []
    for i in range(min(nbr, len(pop))):
        pi = selection(np.delete(pop_indices, parents_i), pool_size)
        parents_i.append(pi)
    parents = [pop[i] for i in parents_i]
    print("Parents: (", end="")
    print(*[pop[i].get_max_col() for i in parents_i], sep=", ", end=")\n")
    return parents


def crossover(prob, sols, k_col, local_search, ls_args):
    print("Crossover pillars")
    if len(sols) < 2:
        print("Not enough parents!")
        return None

    common_base = sols[0].copy()
    for p in sols[1:]:
        common_base[common_base[:] != p[:]] = 0
    common_base[common_base[:] >= k_col] = 0
    print("percent of pillars:", np.sum(common_base[:] != 0)/prob.v_size)
    child = local_search(prob, sol=common_base, start_col=k_col, **ls_args)
    return child


def update_population(prob, pop, eval_func):
    print("Update")
    sum_val = []
    pcol_val = []
    for s in pop:
        sum_val.append(eval_func(prob, s))
        pcol_val.append(s.get_max_col())
    order = np.lexsort((np.array(sum_val), np.array(pcol_val)))
    print(np.array([[i, j] for i, j in zip(pcol_val, sum_val)])[order])
    pop = [pop[i] for i in order]
    return pop


def memetic_algorithm(prob, pop_size, nbr_gen, pool_size, p_nbr,
                      local_search, ls_args, init_heur, init_args, eval_func):

    pop = generate_population3(prob, pop_size, init_heur, init_args)
    # for i, indiv in enumerate(pop):
    #     pop[i] = local_search(prob, sol=indiv, **ls_args)
    #     print("individu #", i, "'s quality:", indiv.get_max_col())
    pop = update_population(prob, pop, eval_func)

    best_sol = pop[0]
    best_score = best_sol.get_max_col()
    for i in range(nbr_gen):
        print("############### generation", i, "################")
        # print("generation #", i)
        parents = choose_parents(pop, p_nbr, pool_size)
        child = crossover(prob, parents, best_sol.get_max_col()-1,
                          local_search, ls_args)
        pop.append(child)
        pop = update_population(prob, pop, eval_func)
        pop = pop[:pop_size]

        if pop[0].get_max_col() < best_score:
            best_sol = pop[0]
            best_score = best_sol.get_max_col()

        print("")

    return best_sol
