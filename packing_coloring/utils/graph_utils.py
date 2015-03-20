# -*- coding: utf-8 -*-

import graph_tool.all as gt
import numpy as np

def graph_from_adj(adj_matrix):
    g = gt.Graph(directed=False)
    g.add_vertex(adj_matrix.shape[0])

    edges = np.transpose(np.nonzero(np.triu(adj_matrix)))
    for e in edges:
        ed = g.add_edge(g.vertex(e[0]), g.vertex(e[1]))
    return g

def graph_from_dist(dist_matrix):
    g = gt.Graph(directed=False)
    adj_matrix = np.zeros_like(dist_matrix)
    adj_matrix[dist_matrix == 1] = 1
    g.add_vertex(adj_matrix.shape[0])

    edges = np.transpose(np.nonzero(np.triu(adj_matrix)))
    for e in edges:
        ed = g.add_edge(g.vertex(e[0]), g.vertex(e[1]))
    return g

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