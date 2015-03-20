# -*- coding: utf-8 -*-

from packing_coloring.utils.graph_utils import *
import numpy as np

def is_valid_coloring(g, colorMap):
    adj = get_adjacency_matrix(g)

    if adj.shape != (len(colorMap), len(colorMap)):
        return False

    for i,c in enumerate(colorMap):
        neighborhood_color = np.ravel(adj[i]) * colorMap
        has_same_color = (neighborhood_color == c)
        if np.any(has_same_color):
            return False

    return True

def is_valid_packing_coloring(g, colorMap):
    dist_matrix = get_distance_matrix(g)

    color_min = np.min(colorMap)
    color_max = np.max(colorMap)

    if g.num_vertices() != len(colorMap):
        return False
    elif color_min <= 0 or color_max <= 0 or color_max < color_min:
        return False

    for c in np.arange(color_min, color_max+1):
        has_color_c = (colorMap == c)
        c_colored_dist = dist_matrix[np.ix_(has_color_c, has_color_c)]
        dist_lq_c = (c_colored_dist <= c)
        dist_lq_c = np.logical_and(dist_lq_c, (np.diag(np.diag(dist_lq_c)) == False))
        if np.any(dist_lq_c):
            return False

    return True
