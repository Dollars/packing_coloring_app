import numpy as np
import numpy.random as rd
import time

from packing_coloring.algorithms.search_space.complete_illegal_col import *
from packing_coloring.algorithms.search_space.partial_valide_col import *
from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm
from packing_coloring.algorithms.constructive.swo_algo import swo_algorithm


def generate_population(prob, heuristic, size, **kwargs):
    pop = []
    for i in range(size):
        print(i)
        indiv = heuristic(prob, **kwargs)
        pop.append(indiv)
    return pop


def choose_parents(pop):
    p1_i, p2_i = rd.choice(np.arange(len(pop)), 2, replace=False)
    parent1 = pop[0]
    parent2 = pop[p2_i]
    return parent1, parent2


def crossover(prob, sol1, sol2):
    child = PackColSolution(prob)
    diam = prob.get_diam()
    max_pack = min(diam, sol1.get_max_col(), sol2.get_max_col())
    sol1_packing = sol1.get_partition()[:max_pack]
    sol2_packing = sol2.get_partition()[:max_pack]

    for i in range(max_pack):
        sol1_card = np.sum(sol1_packing, axis=1)
        sol2_card = np.sum(sol2_packing, axis=1)
        sol_card = np.add(sol1_card, sol2_card)

        pack_i = -1
        new_packing = None
        if (i % 2) == 1:
            sol1_card = np.nan_to_num(np.divide(sol1_card, sol_card))
            pack_i = np.argsort(sol1_card)[-1]
            new_packing = sol1_packing[pack_i]
            sol2_packing[pack_i, :] = 0
        else:
            sol2_card = np.nan_to_num(np.divide(sol2_card, sol_card))
            pack_i = np.argsort(sol2_card)[-1]
            new_packing = sol2_packing[pack_i]
            sol1_packing[pack_i, :] = 0

        child[new_packing == 1] = pack_i+1
        sol1_packing[..., new_packing == 1] = 0
        sol2_packing[..., new_packing == 1] = 0

    if np.any(child == 0):
        child = rlf_algorithm(prob, child)

    return child


def update_population(prob, eval_func, pop):
    sort_val = []
    for s in pop:
        sort_val.append(eval_func(s))
    sort_val = np.argsort(np.array(sort_val))
    pop = [pop[i] for i in sort_val]
    pop.pop()
    return pop


def hybrid_algorithm(prob, local_search, pop_size, init_heur, eval_func, max_iter=500):
    pop = generate_population(prob, init_heur, pop_size+1)
    pop = update_population(prob, eval_func, pop)

    best_sol = pop[0]
    best_score = eval_func(best_sol)

    while max_iter > 0:
        print("generation #", max_iter, flush=True)
        parent1, parent2 = choose_parents(pop)
        child = crossover(prob, parent1, parent2)
        child = local_search(prob, sol=child, duration=5)
        pop = update_population(prob, eval_func, pop)

        if eval_func(pop[0]) < best_score:
            best_sol = pop[0]
            best_score = eval_func(best_sol)

        max_iter -= 1

    return best_sol
