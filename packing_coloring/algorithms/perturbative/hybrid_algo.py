import numpy as np
import numpy.random as rd
import time

from packing_coloring.algorithms.search_space.complete_illegal_col import *
from packing_coloring.algorithms.solution import *
from packing_coloring.algorithms.problem import *
from packing_coloring.algorithms.perturbative.tabupackcol import *
from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm
from packing_coloring.algorithms.constructive.greedy_algo import greedy_algorithm
from packing_coloring.utils.benchmark_utils import set_env, search_step_trace


def generate_population(prob, size, heuristic, init_args):
    print("Init population by permut")
    pop = []
    indiv = heuristic(prob, random_init=False, **init_args)
    permut = np.arange(1, indiv.get_max_col()+1, dtype=int)
    pop.append(indiv)

    # print("init candidate #", 0, ":", indiv.get_max_col(), flush=True)
    indiv.get_trace()
    search_step_trace.clear_all()

    for i in range(1, size):
        new_permut = rd.permutation(permut)
        priority = indiv.get_by_permut(new_permut)
        new_indiv = greedy_algorithm(prob, priority)
        pop.append(new_indiv)

        # print("init candidate #", i, ":", pop[i].get_max_col(), flush=True)        
        new_indiv.get_trace()
        search_step_trace.clear_all()

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


@search_step_trace
def selection(pop, tournament_size=2):
    # Tournament Selection
    print("Tournament ", end="")
    indices = np.arange(len(pop), dtype=int)
    best = rd.choice(indices, 1)[0]

    for i in range(tournament_size):
        adv = rd.choice(np.delete(indices, best), 1)[0]
        if pop[adv] < pop[best]:
            best = adv
    print(best)
    return pop[best]


@search_step_trace
def choose_parents(pop, nbr, tournament_size):
    print("Parents selection")
    pop_indices = np.arange(len(pop), dtype=int)
    parents_i = []
    for i in range(min(nbr, len(pop))):
        pi = selection(np.delete(pop_indices, parents_i), tournament_size)
        parents_i.append(pi)
    parents = [pop[i] for i in parents_i]
    return parents


@search_step_trace
def crossover(prob, sols):
    print("Crossover stupid and easy")
    p1 = sols[0].get_greedy_order()
    p2 = sols[1].get_greedy_order()
    child_permut = p1
    chrom_size = np.floor(len(child_permut)/2)
    to_change = rd.choice(np.arange(prob.v_size), chrom_size, replace=False)
    deleted = child_permut[to_change]
    replacing = np.intersect1d(p2, deleted)
    child_permut[to_change] = replacing
    return greedy_algorithm(prob, child_permut)


@search_step_trace
def crossover_cx(prob, sols):
    print("Crossover cycle")
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


@search_step_trace
def crossover_cover(prob, sols):
    print("Crossover cover")
    diff_rate = np.sum(np.equal(sols[0][:], sols[1][:]))/prob.v_size
    child = PackColSolution(prob)
    diam = prob.get_diam()
    max_pack = min(diam-1, sols[0].get_max_col(), sols[1].get_max_col()) - 1
    max_pack = int(np.ceil(max_pack * (1. - diff_rate)))
    sol1_packing = sols[0].get_partitions()[:max_pack]
    sol2_packing = sols[1].get_partitions()[:max_pack]

    for i in range(max_pack-1):
        new_packing = None
        if (i % 2) == 1:
            sol_packing = sol1_packing
        else:
            sol_packing = sol2_packing

        scores = np.zeros(max_pack, dtype=float)
        for col in np.arange(1, max_pack):
            kcol_nodes = sol_packing[col]
            dist_mat = prob.dist_matrix[kcol_nodes]
            cover_score = np.sum(dist_mat <= col)
            cover_score -= np.sum(kcol_nodes)
            scores[col] = cover_score

        new_col = np.argmax(scores)
        # scores[scores == 0] = float("inf")
        # new_col = np.argmin(scores)
        new_packing = np.copy(sol_packing[new_col])
        child[new_packing] = new_col

        sol1_packing[..., new_packing] = False
        sol2_packing[..., new_packing] = False
        sol1_packing[new_col] = False
        sol2_packing[new_col] = False

    if np.any(child == 0):
        child = rlf_algorithm(prob, child)
    print(child.get_area_score(prob))

    if count_conflicting_edge(prob, child) > 0:
        print("Fail !")

    return child


@search_step_trace
def crossover_area(prob, sols):
    print("Crossover area")
    child = PackColSolution(prob)
    diam = prob.get_diam()
    diff_rate = np.sum(np.equal(sols[0][:], sols[1][:]))/prob.v_size
    max_pack = min(diam-1, sols[0].get_max_col(), sols[1].get_max_col()) - 1
    max_pack = int(np.ceil(max_pack * diff_rate))
    sol1_packing = sols[0].get_partitions()[:max_pack]
    sol2_packing = sols[1].get_partitions()[:max_pack]

    print(diff_rate)
    for i in range(max(2, max_pack-1)):
        new_packing = None
        sol_packing = None
        if (i % 2) == 0:
            sol_packing = sol1_packing
        else:
            sol_packing = sol2_packing

        if np.sum(sol_packing) <= 0:
            break

        scores = np.zeros(max_pack, dtype=float)
        for col in np.arange(1, max_pack-1):
            kcol_nodes = sol_packing[col]
            if np.sum(kcol_nodes) > 0:
                dist_mat = prob.dist_matrix.A
                kcol_dist = dist_mat[kcol_nodes]

                first_half = np.floor(float(col)/2)
                half_nodes = kcol_dist <= first_half
                # half_nodes[dist_mat == 0] = False
                area_score = np.sum(half_nodes)
                if col % 2 == 1:
                    border = np.ceil(float(col)/2)
                    for x in np.arange(prob.v_size)[kcol_nodes]:
                        x_dist = dist_mat[x]
                        x_half_nodes = (x_dist <= first_half)
                        border_nodes = (x_dist == border)
                        for y in np.arange(prob.v_size)[border_nodes]:
                            y_neighbors = (dist_mat[y] == 1)
                            common = np.logical_and(y_neighbors, x_half_nodes)
                            area_score += (float(np.sum(common)) /
                                           np.sum(y_neighbors))
                # area_score = area_score/np.sum(kcol_nodes)
                scores[col] = area_score

        new_col = np.argmax(scores)
        # scores[scores == 0] = float("inf")
        # new_col = np.argmin(scores)
        new_packing = np.copy(sol_packing[new_col])
        child[new_packing] = new_col

        sol1_packing[..., new_packing] = False
        sol2_packing[..., new_packing] = False
        sol1_packing[new_col] = False
        sol2_packing[new_col] = False

    # colors = np.copy(np.unique(child[:]))
    # for i, j in enumerate(colors):
    #     if i < j:
    #         child[child == j] = i

    if np.any(child == 0):
        child = rlf_algorithm(prob, child)

    if count_conflicting_edge(prob, child) > 0:
        print("Fail !")

    return child


@search_step_trace
def mutation(prob, sol, local_search, ls_args):
    print("Mutation")
    diam = prob.get_diam()
    bounds = np.zeros(prob.v_size, dtype=int)
    for i in range(sol.v_size):
        i_col = min(sol[i], diam - 1) + 1
        bound = np.sum(prob.dist_matrix[i] == i_col)
        bounds[i] = bound

    v = np.argmax(bounds)
    v_col = min(sol[v], diam - 1) + 1
    adj_mat = (prob.dist_matrix == 1)
    # changes = (prob.dist_matrix[v] == v_col)
    changes = np.logical_or((prob.dist_matrix[v] == v_col), adj_mat[v])
    adj_mat[v] = changes
    adj_mat[..., v] = np.transpose(changes)
    new_prob = GraphProblem(adj_mat)
    new_sol = PackColSolution(new_prob)
    new_sol = rlf_algorithm(new_prob)
    new_sol = local_search(new_prob, sol=new_sol, **ls_args)

    # mutated = PackColSolution(prob)
    # mutated[v] = new_sol[v]
    # mutated = rlf_algorithm(prob, sol=mutated)

    ordering = new_sol.get_greedy_order()
    mutated = greedy_algorithm(prob, ordering)
    print("mutation diff:", np.sum(np.equal(mutated[:], sol[:]))/prob.v_size)
    return mutated


@search_step_trace
def update_population(prob, pop, eval_func, nbr_gen=None):
    print("Update")
    sum_val = []
    pcol_val = []
    for s in pop:
        sum_val.append(eval_func(prob, s))
        pcol_val.append(s.get_max_col())
    order = np.lexsort((np.array(sum_val), np.array(pcol_val)))
    print(np.array([[i, j] for i, j in zip(pcol_val, sum_val)])[order])
    pop = [pop[i] for i in order]

    if nbr_gen is not None:
        tracefname = "hybrid_algorithm.qst"
        with open(tracefname, 'a') as f:
            for indiv in pop:
                print(prob.name, ", ", indiv.get_max_col(), ", ",
                      nbr_gen, file=f, sep="")
    return pop


def hybrid_algorithm(prob, pop_size, nbr_gen, pool_size, replace_rate, mut_prob,
                     local_search, ls_args, init_heur, init_args, eval_func):
    pop = generate_population3(prob, pop_size, init_heur, init_args)
    # for i, indiv in enumerate(pop):
    #     pop[i] = local_search(prob, sol=indiv, **ls_args)
        # print("individu #", i, "'s quality:", pop[i].get_max_col())

    pop = update_population(prob, pop, eval_func, 0)

    best_sol = pop[0]
    best_score = best_sol.get_max_col()
    search_step_trace.print_trace(prob, best_sol)

    new_gen_size = np.ceil(pop_size * replace_rate)
    for gen in range(nbr_gen):
        print("############### generation", gen, "################")
        new_gen = []
        while len(new_gen) < new_gen_size:
            parents = choose_parents(pop, 2, pool_size)
            print("Parents: (", parents[0].get_max_col(), ", ", parents[1].get_max_col(),")", sep="")
            child = crossover_area(prob, parents)
            # child = crossover_area(prob, parents)
            # child = crossover_cover(prob, parents)
            print("Resulting child", child.get_max_col(), end="")
            child = local_search(prob, sol=child, **ls_args)
            print(" ->", child.get_max_col())
            new_gen.append(child)

        new_gen = update_population(prob, new_gen, eval_func)
        if new_gen[0].get_max_col() < best_score:
            best_sol = new_gen[0].copy()
            best_score = best_sol.get_max_col()
            search_step_trace.print_trace(prob, best_sol)

        for i, indiv in enumerate(new_gen):
            if rd.rand() < mut_prob:
                print("Before mutation", indiv.get_max_col())
                indiv = mutation(prob, indiv, local_search, ls_args)
                print("After mutation", indiv.get_max_col())
                indiv = local_search(prob, sol=indiv, **ls_args)
                print("Result", indiv.get_max_col())
                print("")
                new_gen[i] = indiv

        pop = new_gen + pop[:(pop_size - len(new_gen))]
        pop = update_population(prob, pop, eval_func, gen+1)

        if pop[0].get_max_col() < best_score:
            best_sol = pop[0].copy()
            best_score = best_sol.get_max_col()
            search_step_trace.print_trace(prob, best_sol)

        print("")

    return best_sol
