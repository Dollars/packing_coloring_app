import graph_tool.all as gt
# import numpy as np
import sys
# import json

import yappi

# import packing_coloring.graph_generator as gg
sys.path.append('..')
from packing_coloring.utils import get_distance_matrix
from packing_coloring.algorithms.problem import GraphProblem
from packing_coloring.algorithms.perturbative.tabupackcol import partial_pack_col, react_partial_pack_col
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

    for i in range(nbr_it):
        sol = func(prob, *func_args, **func_kwargs)
        search_step_trace.clear_all()
        if sol.get_max_col() < best_score:
            best_sol = sol

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


def square_grid_pts_pcoloring(size, nbr_it):
    g = gt.lattice([size, size])
    kwargs = {"k_count": 10, "tt_a": 10, "tt_d": 0.5,
              "max_iter": 2000, "count_max": 5}
    return benchmark_function(g, nbr_it, partial_pack_col, **kwargs)


def square_grid_rps_pcoloring(size, nbr_it):
    g = gt.lattice([size, size])
    kwargs = {"k_count": 5, "tt_a": 10, "tt_d": 0.6,
              "max_iter": 20000, "iter_period": 50, "tenure_inc": 5}
    return benchmark_function(g, nbr_it, react_partial_pack_col, **kwargs)


sol = square_grid_pts_pcoloring(24, 20)
print(sol)

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
