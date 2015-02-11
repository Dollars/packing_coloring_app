# -*- coding: utf-8 -*-

import pytest

from utils import *
import graph_tool.all as gt

def test_is_valid_coloring():
    g = gt.load_graph("Test/star_shape_pcolored.gt")
    colorMap = g.vertex_properties["colorMap"]
    assert is_valid_coloring(g, colorMap.a)

    g = gt.load_graph("Test/star_shape_colored.gt")
    colorMap = g.vertex_properties["colorMap"]
    result = is_valid_coloring(g, colorMap.a)
    assert result

def test_is_valid_packing_coloring():
    g = gt.load_graph("Test/star_shape_pcolored.gt")
    colorMap = g.vertex_properties["colorMap"]
    result = is_valid_packing_coloring(g, colorMap.a)
    assert result

    g = gt.load_graph("Test/star_shape_colored.gt")
    colorMap = g.vertex_properties["colorMap"]
    result = is_valid_packing_coloring(g, colorMap.a)
    assert result == False
