#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: dollars
# @Date:   2014-11-30 13:28:08

from graph_tool import Graph
import re


def parse_col(filename):
    g = Graph(directed=False)
    f = open(filename)
    start_zero = False

    for line in f:
        if line[0] == 'c' or line[0] == '\n':
            continue
        elif line[0] == 'p':
            ve_nbr = re.match("p edge ([-+]?\d+) ([-+]?\d+)", line)
            v_nbr = int(ve_nbr.group(1))
            e_nbr = int(ve_nbr.group(2))
            g.add_vertex(v_nbr+1)
        elif line[0] == 'e':
            edges = re.finditer("e ([-+]?\d+) ([-+]?\d+)", line)
            for edge in edges:
                a = int(edge.group(1))
                b = int(edge.group(2))
                g.add_edge(g.vertex(a), g.vertex(b))
                if a == 0 or b == 0:
                    start_zero = True
        else:
            assert False

    if start_zero:
        g.remove_vertex(v_nbr)
    else:
        g.remove_vertex(0)

    return g
