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
    for i in range(size):
        print("init candidate #", i, flush=True)
        indiv = heuristic(prob, random_init=((i % 2) == 0), **kwargs)
        pop.append(indiv)
    return pop


def choose_parents(pop):
    p1_i, p2_i = rd.choice(np.arange(len(pop)), 2, replace=False)
    parent1 = pop[p1_i]
    parent2 = pop[p2_i]
    return parent1, parent2


def crossover2(prob, sol1, sol2):
    p1 = sol1.get_greedy_order()
    p2 = sol2.get_greedy_order()
    child = p1
    chrom_size = len(child)
    to_change = rd.choice(np.arange(chrom_size), np.floor(chrom_size/2), replace=False)
    deleted = child[to_change]
    replacing = np.intersect1d(p2, deleted)
    child[to_change] = replacing
    return greedy_algorithm(prob, child)
    # return sol1


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


def mutation(prob, sol, heuristic, local_search):
    bounds = np.zeros(prob.v_size, dtype=int)
    for i in range(sol.v_size):
        i_col = sol[i]
        bound = np.sum(prob.dist_matrix[i] == i_col)
        bounds[i] = bound

    v = np.argmax(bounds)
    adj_mat = (prob.dist_matrix == 1)
    changes = (prob.dist_matrix[v] == sol[v])
    adj_mat[v] = changes
    adj_mat[..., v] = np.transpose(changes)
    new_prob = GraphProblem(adj_mat)
    new_sol = heuristic(new_prob, random_init=False)
    new_sol = local_search(new_prob, sol=new_sol, duration=5)
    return greedy_algorithm(prob, new_sol.get_greedy_order())


def update_population(prob, eval_func, pop):
    sum_val = []
    pcol_val = []
    for s in pop:
        sum_val.append(eval_func(s))
        pcol_val.append(s.get_max_col())
    order = np.lexsort((np.array(sum_val), np.array(pcol_val)))
    print( np.array([[i, j] for i, j in zip(pcol_val, sum_val)])[order] )
    pop = [pop[i] for i in order]
    return pop


def hybrid_algorithm(prob, local_search, pop_size, init_heur, eval_func, max_iter=500):
    pop = generate_population(prob, init_heur, pop_size)
    for i, indiv in enumerate(pop):
        pop[i] = local_search(prob, sol=indiv, duration=5)

    pop = update_population(prob, eval_func, pop)

    best_sol = pop[0]
    best_score = eval_func(best_sol)

    while max_iter > 0:
        print("generation #", max_iter, flush=True)
        parent1, parent2 = choose_parents(pop)
        child = crossover(prob, parent1, parent2)
        child = local_search(prob, sol=child, duration=5)
        pop.append(child)
        child1 = mutation(prob, child, init_heur, local_search)
        child1 = local_search(prob, sol=child1, duration=5)
        pop.append(child1)
        pop = update_population(prob, eval_func, pop)
        pop = pop[:pop_size]

        if eval_func(pop[0]) < best_score:
            best_sol = pop[0]
            best_score = eval_func(best_sol)

        max_iter -= 1

    return best_sol
