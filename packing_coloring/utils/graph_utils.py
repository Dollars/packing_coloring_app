# -*- coding: utf-8 -*-

import graph_tool.all as gt
import numpy as np

def get_distance_matrix(g):
    dist = gt.shortest_distance(g)
    dim = g.num_vertices()
    m_dist = np.zeros((1, dim), dtype=int)
    for v in g.vertices():
        m_dist = np.append(m_dist, np.array(dist[v], ndmin=2), axis=0)
    m_dist = np.delete(m_dist, 0, 0)
    return m_dist

def get_adjacency_matrix(g):
    g_adj = gt.adjacency(g)
    return g_adj.todense()
