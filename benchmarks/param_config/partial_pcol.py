#!/usr/bin/env python

import graph_tool.all as gt
import numpy as np
import sys

# import packing_coloring.graph_generator as gg
sys.path.append('../../..')
import packing_coloring.utils.benchmark_utils as bu
bu.GLOBAL_ENABLE_FLAG = True
from packing_coloring.utils import get_distance_matrix
import packing_coloring.algorithms.search_space.partial_valide_col as pvc
import packing_coloring.graph_generator.generator as gntr
from packing_coloring.algorithms.problem import GraphProblem
from packing_coloring.algorithms.perturbative.tabupackcol import partial_pack_col
import argparse

pvc.random_ok = False
np.random.seed(10)


def benchmark_function(graphe, func, *func_args, **func_kwargs):
    dist_mat = get_distance_matrix(graphe)
    prob = GraphProblem(dist_mat)

    if hasattr(graphe, "name"):
        prob.name = graphe.name

    sol = func(prob, *func_args, **func_kwargs)

    return sol

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('instance',
                        type=str)
    parser.add_argument('--k_count', dest='k_count',
                        type=int, required=True)
    parser.add_argument('--tt_a', dest='tt_a',
                        type=int, required=True)
    parser.add_argument('--tt_d', dest='tt_d',
                        type=float, required=True)
    parser.add_argument('--max_iter', dest='max_iter',
                        type=int, required=True)
    parser.add_argument('--count_max', dest='count_max',
                        type=int, required=True)
    args = parser.parse_args()

    g = gt.load_graph(args.instance)
    kwargs = {"k_count": args.k_count,
              "tt_a": args.tt_a,
              "tt_d": args.tt_d,
              "max_iter": args.max_iter,
              "count_max": args.max_iter}

    sol = benchmark_function(g, partial_pack_col, **kwargs)
    print(sol.get_max_col())
