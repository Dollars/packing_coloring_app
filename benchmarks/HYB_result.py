import graph_tool.all as gt
# import numpy as np
import sys
# import json

# import packing_coloring.graph_generator as gg
sys.path.append('..')
from packing_coloring.utils import get_distance_matrix
from packing_coloring.graph_generator.generator import distance_graph
from packing_coloring.algorithms.problem import GraphProblem
from packing_coloring.algorithms.constructive.swo_algo import swo_algorithm
# from packing_coloring.algorithms.constructive.rlf_algo import rlf_algorithm
from packing_coloring.algorithms.perturbative.tabupackcol import partial_pack_col, react_partial_pack_col
from packing_coloring.algorithms.perturbative.hybrid_algo import hybrid_algorithm
from packing_coloring.utils.benchmark_utils import search_step_trace
# from packing_coloring.utils.benchmark_utils import YProfiler
# from packing_coloring.algorithms.search_space.partial_valide_col import best_i_swap, assign_col

# profiler = YProfiler(partial_pack_col, partial_kpack_col)
# partial_pack_col = profiler.wrap_function(partial_pack_col)


def benchmark_function(graphe, nbr_it, func, *func_args, **func_kwargs):
    best_sol = None
    best_score = float("inf")

    dist_mat = get_distance_matrix(graphe)
    prob = GraphProblem(dist_mat)

    if hasattr(graphe, "name"):
        prob.name = graphe.name

    for i in range(nbr_it):
        print(func.__name__, "#", i)

        sol = func(prob, *func_args, **func_kwargs)
        if sol.get_max_col() < best_score:
            best_sol = sol
            best_score = best_sol.get_max_col()
        search_step_trace.clear_all()

        tracefname = "{0}.qst".format(func.__name__)
        with open(tracefname, 'a') as f:
            for name, data in sol.record.items():
                print(prob.name, ", ", sol.get_max_col(), ", ",
                      search_step_trace.csv_format().format(name, data),
                      file=f, sep="")
            print("", file=f)

    solfname = "{0}_{1}.pcol".format(prob.name, best_sol.get_max_col())
    with open(solfname, 'a') as f:
        print(best_sol, file=f)
        print("", file=f)

    return best_sol


def square_grid_partcol_pcoloring(size, nbr_it):
    g = gt.lattice([size, size])
    setattr(g, "name", "G {0}x{0}".format(size))
    kwargs = {"k_count": 10, "tt_a": 20, "tt_d": 0.6,
              "max_iter": 2000, "count_max": 5}
    return benchmark_function(g, nbr_it, partial_pack_col, **kwargs)


def dist_graph_hyba_pcoloring(k, t, size, nbr_it):
    g = distance_graph((k, t), size)
    setattr(g, "name", "D({0},{1})".format(k, t))

    eval_func = lambda prob, a: a.get_area_score(prob)
    ls_args = {"k_count": 3, "tt_a": 50, "tt_d": 0.6,
               "max_iter": 2000, "count_max": 5}
    init_args = {"iter_count": 50, "blame_value": 25, "blame_rate": 0.85}

    kwargs = {"pop_size": 5, "nbr_gen": 5, "pool_nbr": 1,
              "replace_rate": 0.4, "mut_prob": 0.2,
              "local_search": partial_pack_col, "ls_args": ls_args,
              "init_heur": swo_algorithm, "init_args": init_args,
              "eval_func": eval_func}

    return benchmark_function(g, nbr_it, hybrid_algorithm, **kwargs)


def square_grid_rps_pcoloring(size, nbr_it):
    g = gt.lattice([size, size])
    kwargs = {"k_count": 5, "tt_a": 20, "tt_d": 0.6,
              "max_iter": 20000, "iter_period": 50, "tenure_inc": 5}
    return benchmark_function(g, nbr_it, react_partial_pack_col, **kwargs)


# --- Grid/lattice graphes --- #
# square_grid_partcol_pcoloring(24, 20)


# --- Distance graphes --- #
# Packing chromatic number already met
# dist_graph_hyba_pcoloring(2, 3, 240, 30)
# dist_graph_hyba_pcoloring(3, 5, 192, 30)
# dist_graph_hyba_pcoloring(1, 4, 320, 10)
# dist_graph_hyba_pcoloring(1, 5, 1028, 10)

# Small sized instances

# dist_graph_hyba_pcoloring(1, 6, 2016, 10)
# dist_graph_hyba_pcoloring(1, 7, 640, 10)
# dist_graph_hyba_pcoloring(1, 8, 5184, 10)
# dist_graph_hyba_pcoloring(1, 9, 576, 10)

# dist_graph_hyba_pcoloring(2, 5, 336, 10)
# dist_graph_hyba_pcoloring(2, 7, 2376, 10)
# dist_graph_hyba_pcoloring(2, 9, 4224, 10)

# dist_graph_hyba_pcoloring(3, 4, 672, 10)
# dist_graph_hyba_pcoloring(3, 7, 8720, 10)
# dist_graph_hyba_pcoloring(3, 8, 6636, 10)
# dist_graph_hyba_pcoloring(3, 10, 3120, 10)

# dist_graph_hyba_pcoloring(4, 5, 972, 1)
# dist_graph_hyba_pcoloring(4, 7, 7040, 10)
# dist_graph_hyba_pcoloring(4, 9, 3952, 10)

# dist_graph_hyba_pcoloring(5, 6, 1584, 10)
# dist_graph_hyba_pcoloring(5, 7, 768, 10)
dist_graph_hyba_pcoloring(5, 8, 2496, 1)
# dist_graph_hyba_pcoloring(5, 9, 792, 10)

# dist_graph_hyba_pcoloring(6, 7, 1716, 10)

# dist_graph_hyba_pcoloring(7, 8, 6720, 10)
# dist_graph_hyba_pcoloring(7, 9, 768, 10)
# dist_graph_hyba_pcoloring(7, 10, 24480, 10)

# dist_graph_hyba_pcoloring(8, 9, 51751, 10)

# dist_graph_hyba_pcoloring(9, 10, 15048, 10)



# random_pts_pcoloring()

# with open("partial_pcol.prof", 'w') as f:
#     for key, item in profiler.results["partial_pack_col"].items():
#         print('{:#^75}'.format("graph name: %s" % key), file=f)
#         for num, res in item:
#             print(num, file=f)
#             for st in res:
#                 test = yappi.YFuncStat(list(st.values()))
#                 test._print(out=f, columns={0:("name",30), 1:("ncall", 10), 2:("tsub", 8), 3:("ttot", 8), 4:("tavg",8)})
#             print("", file=f)
