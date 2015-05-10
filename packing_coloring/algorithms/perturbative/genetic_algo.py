import numpy as np
import numpy.random as rd
import time

from packing_coloring.algorithms.search_space.complete_illegal_col import *
from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.problem import *
from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm
from packing_coloring.algorithms.constructive.greedy_algo import greedy_algorithm


def generate_population(prob, heuristic, size, **kwargs):
    pop = []
    indiv = heuristic(prob, random_init=False, **kwargs)
    permut = np.arange(1, indiv.get_max_col()+1, dtype=int)
    pop.append(indiv)
    print("init candidate #", 0, ":", indiv.get_max_col(), flush=True)
    for i in range(1, size):
        new_permut = rd.permutation(permut)
        priority = indiv.get_by_permut(new_permut)
        pop.append(greedy_algorithm(prob, priority))
        print("init candidate #", i, ":", pop[i].get_max_col(), flush=True)
    return pop


def generate_population2(prob, heuristic, size, **kwargs):
    pop = []
    for i in range(size):
        print("init candidate #", i, flush=True)
        indiv = heuristic(prob, random_init=(i != 0), **kwargs)
        pop.append(indiv)
    return pop


def selection(pop, tournament_size=1):
    # Tournament Selection
    indices = np.arange(len(pop), dtype=int)
    best = rd.choice(indices, 1)[0]

    for i in range(tournament_size-1):
        adv = rd.choice(np.delete(indices, best), 1)[0]
        if pop[adv] < pop[best]:
            best = adv
    return pop[best]


def choose_parents(pop):
    pop_indices = np.arange(len(pop), dtype=int)
    p1_i = selection(pop_indices, 1)
    p2_i = selection(np.delete(pop_indices, p1_i), 1)
    parent1 = pop[p1_i]
    parent2 = pop[p2_i]
    print("parent1:", p1_i, "->", parent1.get_max_col())
    print("parent2:", p2_i, "->", parent2.get_max_col())
    return parent1, parent2


def crossover(prob, sol1, sol2, local_search):
    child = PackColSolution(prob)
    pillars = (sol1 == sol2)
    child[pillars] = sol1[pillars]
    if np.any(child == 0):
        child = rlf_algorithm(prob, child)
    print("child:", child.get_max_col())
    child = local_search(prob, sol=child, k_count=3, tt_a=20, tt_d=0.6, max_iter=1000, pillars=pillars)
    print("child improved:", child.get_max_col())
    return child


def crossover3(prob, sol1, sol2):
    p1 = sol1.get_greedy_order()
    p2 = sol2.get_greedy_order()
    child = p1
    chrom_size = len(child)
    to_change = rd.choice(np.arange(chrom_size), np.floor(chrom_size/2), replace=False)
    deleted = child[to_change]
    replacing = np.intersect1d(p2, deleted)
    child[to_change] = replacing
    return greedy_algorithm(prob, child)


def crossover2(prob, sol1, sol2):
    child = PackColSolution(prob)
    diam = prob.get_diam()
    max_pack = min(diam-1, sol1.get_max_col(), sol2.get_max_col())
    sol1_packing = sol1.get_partitions()[:max_pack]
    sol2_packing = sol2.get_partitions()[:max_pack]

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


def mutation(prob, sol, local_search):
    diam = prob.get_diam()
    bounds = np.zeros(prob.v_size, dtype=int)
    for i in range(sol.v_size):
        i_col = min(sol[i], diam - 1) + 1
        bound = np.sum(prob.dist_matrix[i] == i_col)
        bounds[i] = bound

    v = np.argmax(bounds)
    v_col = min(sol[v], diam - 1) + 1
    adj_mat = (prob.dist_matrix == 1)
    changes = (prob.dist_matrix[v] == v_col)
    adj_mat[v] = changes
    adj_mat[..., v] = np.transpose(changes)
    new_prob = GraphProblem(adj_mat)
    new_sol = rlf_algorithm(new_prob)
    new_sol = local_search(new_prob, sol=new_sol, duration=5)
    return greedy_algorithm(prob, new_sol.get_greedy_order())


def update_population(prob, eval_func, pop):
    sum_val = []
    pcol_val = []
    for s in pop:
        sum_val.append(eval_func(s))
        pcol_val.append(s.get_max_col())
    order = np.lexsort((np.array(sum_val), np.array(pcol_val)))
    print(np.array([[i, j] for i, j in zip(pcol_val, sum_val)])[order])
    pop = [pop[i] for i in order]
    return pop


def hybrid_algorithm(prob, local_search, pop_size, init_heur, eval_func, max_iter=500):
    pop = generate_population(prob, init_heur, pop_size)
    for i, indiv in enumerate(pop):
        pop[i] = local_search(prob, sol=indiv, duration=5)
        print("individu #", i, "'s quality:", pop[i].get_max_col())

    pop = update_population(prob, eval_func, pop)

    best_sol = pop[0]
    best_score = eval_func(best_sol)

    while max_iter > 0:
        print("generation #", max_iter, flush=True)
        parent1, parent2 = choose_parents(pop)
        child = crossover(prob, parent1, parent2, local_search)
        # print("child:", child.get_max_col())
        # child = local_search(prob, sol=child, k_count=3, tt_a=20, tt_d=0.6, max_iter=10000)
        # print("improved child: ", child.get_max_col())
        pop.append(child)
        child1 = mutation(prob, child, local_search)
        print("mutated child: ", child1.get_max_col())
        child1 = local_search(prob, sol=child1, k_count=3, tt_a=20, tt_d=0.6, max_iter=10000)
        print("improved mutated child: ", child1.get_max_col())
        pop.append(child1)
        pop = update_population(prob, eval_func, pop)
        pop = pop[:pop_size]

        if eval_func(pop[0]) < best_score:
            best_sol = pop[0]
            best_score = eval_func(best_sol)

        print("")
        max_iter -= 1

    return best_sol
