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
    pop = []
    indiv = heuristic(prob, random_init=False, **init_args)
    permut = np.arange(1, indiv.get_max_col()+1, dtype=int)
    pop.append(indiv)
    print("init candidate #", 0, ":", indiv.get_max_col(), flush=True)
    for i in range(1, size):
        new_permut = rd.permutation(permut)
        priority = indiv.get_by_permut(new_permut)
        pop.append(greedy_algorithm(prob, priority))
        print("init candidate #", i, ":", pop[i].get_max_col(), flush=True)
    return pop


def generate_population2(prob, size, heuristic, init_args):
    pop = []
    for i in range(size):
        print("init candidate #", i, flush=True)
        indiv = heuristic(prob, random_init=(i != 0), **init_args)
        pop.append(indiv)
    return pop


def selection(pop, tournament_size=2):
    # Tournament Selection
    indices = np.arange(len(pop), dtype=int)
    best = rd.choice(indices, 1)[0]

    for i in range(tournament_size):
        adv = rd.choice(np.delete(indices, best), 1)[0]
        if pop[adv] < pop[best]:
            best = adv
    return pop[best]


def choose_parents(pop, nbr, tournament_size):
    pop_indices = np.arange(len(pop), dtype=int)
    parents_i = []
    for i in range(min(nbr, len(pop))):
        pi = selection(np.delete(pop_indices, parents_i), tournament_size)
        parents_i.append(pi)
    parents = [pop[i] for i in parents_i]
    return parents


def crossover(prob, sols):
    p1 = sols[0].get_greedy_order()
    p2 = sols[1].get_greedy_order()
    child_permut = p1
    chrom_size = np.floor(len(child_permut)/2)
    to_change = rd.choice(np.arange(prob.v_size), chrom_size, replace=False)
    deleted = child_permut[to_change]
    replacing = np.intersect1d(p2, deleted)
    child_permut[to_change] = replacing
    return greedy_algorithm(prob, child_permut)


def crossover_cx(prob, sols):
    p1 = sols[0].get_greedy_order()
    p2 = sols[1].get_greedy_order()
    child1 = np.zeros(prob.v_size, dtype=int)
    child2 = np.zeros(prob.v_size, dtype=int)

    positions = np.arange(prob.v_size, dtype=int)
    order1 = np.argsort(p1)
    inplace = p1 == p2
    child1[inplace] = p1[inplace]
    child2[inplace] = p2[inplace]
    positions[inplace] = -1

    cycle_nbr = 0
    while np.any(positions != -1):
        cycle = [np.argmax(positions > -1)]

        while p1[cycle[0]] != p2[cycle[-1]]:
            step = p2[cycle[-1]]
            pos = order1[step]
            cycle.append(pos)
        positions[cycle] = -1

        if cycle_nbr % 2 == 0:
            child1[cycle] = p1[cycle]
            child2[cycle] = p2[cycle]
        else:
            child1[cycle] = p2[cycle]
            child2[cycle] = p1[cycle]
        cycle_nbr += 1

    sol1 = greedy_algorithm(prob, child1)
    sol2 = greedy_algorithm(prob, child2)
    if sol1.get_max_col() <= sol2.get_max_col():
        return sol1
    else:
        return sol2


def crossover_cover(prob, sols):
    child = PackColSolution(prob)
    diam = prob.get_diam()
    max_pack = min(diam-1, sols[0].get_max_col(), sols[1].get_max_col())
    sol1_packing = sols[0].get_partitions()[:max_pack+1]
    sol2_packing = sols[1].get_partitions()[:max_pack+1]

    for i in range(max_pack):
        new_packing = None
        if (i % 2) == 1:
            sol_packing = sol1_packing
        else:
            sol_packing = sol2_packing

        scores = np.zeros(max_pack, dtype=int)
        for col in np.arange(1, max_pack):
            kcol_nodes = sol_packing[col]
            dist_mat = prob.dist_matrix[kcol_nodes]
            cover_score = np.sum(dist_mat <= col)
            cover_score -= np.sum(kcol_nodes)
            scores[col] = cover_score

        new_col = np.argmax(scores)
        new_packing = sol_packing[new_col]
        child[new_packing] = new_col

        sol1_packing[..., new_packing == 1] = 0
        sol2_packing[..., new_packing == 1] = 0

    if np.any(child == 0):
        child = rlf_algorithm(prob, child)

    return child


def crossover_area(prob, sols):
    child = PackColSolution(prob)
    diam = prob.get_diam()
    max_pack = min(diam-1, sols[0].get_max_col(), sols[1].get_max_col())
    sol1_packing = sols[0].get_partitions()[:max_pack+1]
    sol2_packing = sols[1].get_partitions()[:max_pack+1]

    for i in range(max_pack):
        new_packing = None
        if (i % 2) == 1:
            sol_packing = sol1_packing
        else:
            sol_packing = sol2_packing

        scores = np.zeros(max_pack, dtype=int)
        for col in np.arange(1, max_pack):
            kcol_nodes = sol_packing[col]
            dist_mat = prob.dist_matrix[kcol_nodes]
            first_half = np.floor(col/2)
            half_nodes = dist_mat <= first_half
            half_nodes[dist_mat == 0] = False
            area_score = np.sum(half_nodes)
            area_nodes = np.sum(half_nodes, axis=0).A1 > 0

            if len(area_nodes) != prob.v_size:
                print("size matters !", len(area_nodes), prob.v_size)

            if col % 2 == 1:
                border = np.ceil(col/2)
                border_nodes = np.sum(dist_mat == border, axis=0).A1 > 0
                for y in np.arange(prob.v_size)[border_nodes]:
                    y_neighbors = prob.dist_matrix[y] == 1
                    common = np.logical_and(y_neighbors, area_nodes)
                    area_score += np.sum(common)/np.sum(y_neighbors)

            scores[col] = area_score

        new_col = np.argmax(scores)
        new_packing = sol_packing[new_col]
        child[new_packing] = new_col

        sol1_packing[..., new_packing == 1] = 0
        sol2_packing[..., new_packing == 1] = 0

    if np.any(child == 0):
        child = rlf_algorithm(prob, child)

    return child


def mutation(prob, sol, local_search, ls_args):
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
    new_sol = local_search(new_prob, sol=new_sol, **ls_args)
    return greedy_algorithm(prob, new_sol.get_greedy_order())


def update_population(prob, pop, eval_func):
    sum_val = []
    pcol_val = []
    for s in pop:
        sum_val.append(eval_func(prob, s))
        pcol_val.append(s.get_max_col())
    order = np.lexsort((np.array(sum_val), np.array(pcol_val)))
    print(np.array([[i, j] for i, j in zip(pcol_val, sum_val)])[order])
    pop = [pop[i] for i in order]
    return pop


def hybrid_algorithm(prob, pop_size, nbr_generation, tournament_size,
                     local_search, ls_args, init_heur, init_args, eval_func):
    pop = generate_population(prob, pop_size, init_heur, init_args)
    for i, indiv in enumerate(pop):
        pop[i] = local_search(prob, sol=indiv, **ls_args)
        print("individu #", i, "'s quality:", pop[i].get_max_col())

    pop = update_population(prob, pop, eval_func)

    best_sol = pop[0]
    best_score = best_sol.get_max_col()

    for i in range(nbr_generation):
        print("generation #", i)
        parents = choose_parents(pop, 2, tournament_size)
        child = crossover_area(prob, parents)
        print("child:", child.get_max_col(), np.sum(child == 0))
        child = local_search(prob, sol=child, **ls_args)
        print("improved child: ", child.get_max_col())
        pop.append(child)

        child1 = mutation(prob, child, local_search, ls_args)
        print("mutated child: ", child1.get_max_col())
        child1 = local_search(prob, sol=child1, **ls_args)
        print("improved mutated child: ", child1.get_max_col())
        pop.append(child1)

        pop = update_population(prob, pop, eval_func)
        pop = pop[:pop_size]

        if pop[0].get_max_col() < best_score:
            best_sol = pop[0]
            best_score = best_sol.get_max_col()

        print("")

    return best_sol
