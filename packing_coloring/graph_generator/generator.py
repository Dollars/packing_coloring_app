# -*- coding: utf-8 -*-

import graph_tool.all as gt
import networkx as nx
from packing_coloring.graph_generator.col_parser import parse_col
import packing_coloring.utils.graph_utils as gu
import numpy as np
import numpy.random as rd


def load_graph_file(file):
    return parse_col(file)


def balanced_tree(r, h):
    gx = nx.balanced_tree(r, h)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def barbell_graph(m1, m2):
    gx = nx.barbell_graph(m1, m2)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def complete_graph(n):
    gx = nx.complete_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def complete_bipartite_graph(n1, n2):
    gx = nx.complete_bipartite_graph(n1, n2)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def circular_ladder_graph(n):
    gx = nx.circular_ladder_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def cycle_graph(n):
    gx = nx.cycle_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def dorogovtsev_goltsev_mendes_graph(n):
    gx = nx.dorogovtsev_goltsev_mendes_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def grid_graph(dim):
    return gt.lattice(dim)


def hypercube_graph(n):
    gx = nx.hypercube_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def ladder_graph(n):
    gx = nx.ladder_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def lollipop_graph(m, n):
    gx = nx.lollipop_graph(m, n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def path_graph(n):
    gx = nx.path_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def star_graph(n):
    gx = nx.star_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def wheel_graph(n):
    gx = nx.wheel_graph(n)
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def bull_graph():
    gx = nx.bull_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def chvatal_graph():
    gx = nx.chvatal_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def cubical_graph():
    gx = nx.cubical_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def desargues_graph():
    gx = nx.desargues_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def diamond_graph():
    gx = nx.diamond_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def dodecahedral_graph():
    gx = nx.dodecahedral_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def frucht_graph():
    gx = nx.frucht_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def heawood_graph():
    gx = nx.heawood_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def house_graph():
    gx = nx.house_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def house_x_graph():
    gx = nx.house_x_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def icosahedral_graph():
    gx = nx.icosahedral_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def krackhardt_kite_graph():
    gx = nx.krackhardt_kite_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def moebius_kantor_graph():
    gx = nx.moebius_kantor_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def octahedral_graph():
    gx = nx.octahedral_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def pappus_graph():
    gx = nx.pappus_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def petersen_graph():
    gx = nx.petersen_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def sedgewick_maze_graph():
    gx = nx.sedgewick_maze_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def tetrahedral_graph():
    gx = nx.tetrahedral_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def truncated_cube_graph():
    gx = nx.truncated_cube_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def truncated_tetrahedron_graph():
    gx = nx.truncated_tetrahedron_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def tutte_graph():
    gx = nx.tutte_graph()
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def distance_graph(steps, size):
    nodes = np.arange(size, dtype=int)
    gx = nx.Graph()
    gx.add_nodes_from(nodes)

    for i in nodes:
        for t in steps:
            v = i + t
            if v < size:
                gx.add_edge(i, v)

    adj_mat = nx.to_numpy_matrix(gx)
    g = gu.graph_from_adj(adj_mat)
    return g


def cylindre_product(m, n):
    gx1 = nx.cycle_graph(m)
    gx2 = nx.cycle_graph(n)
    gx = nx.cartesian_product(gx1, gx2)

    print(gx.number_of_selfloops(), nx.is_connected(gx))
    print(nx.diameter(gx), nx.radius(gx))
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def cylindre_path_product(m, n):
    gx1 = nx.path_graph(m)
    gx2 = nx.cycle_graph(n)
    gx = nx.cartesian_product(gx1, gx2)

    print(gx.number_of_selfloops(), nx.is_connected(gx))
    print(nx.diameter(gx), nx.radius(gx))
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def gnm_random_graph(n, m):
    gx = nx.gnm_random_graph(n, m)
    print(gx.number_of_selfloops(), nx.is_connected(gx))
    print(nx.diameter(gx), nx.radius(gx))
    adj_mat = nx.to_numpy_matrix(gx)

    g = gu.graph_from_adj(adj_mat)
    return g


def geometric_random_graph(size, radius):
    points = rd.random((size, 2)) * 4
    g, pos = gt.geometric_graph(points, radius)

    l = gt.label_largest_component(g)
    g.set_vertex_filter(l)
    g.purge_vertices()
    gt.graph_draw(g, pos=pos, output_size=(300, 300), output="geometric.pdf")

    print(gt.pseudo_diameter(g), )

    return g
