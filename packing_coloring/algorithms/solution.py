# -*- coding: utf-8 -*-

import numpy as np
from packing_coloring.algorithms.problem import *


class PackColSolution:
    def __init__(self, g_prob=None):
        self.pack_col = None
        if g_prob is None:
            self.pack_col = np.zeros([], dtype=int)
        else:
            self.pack_col = np.zeros(g_prob.v_size, dtype=int)
        self._fitness_value = None
        self.evaluated = False

    def copy(self):
        pcol = PackColSolution()
        pcol.pack_col = self.pack_col.copy()
        pcol._fitness_value = self._fitness_value
        pcol.evaluated = self.evaluated
        return pcol

    def evaluate(self, func, prob):
        self._fitness_value = func(prob, sol)
        self.evaluated = True
        return self._fitness_value

    def uncolored(self):
        return self.pack_col == 0

    def colored(self):
        return self.pack_col != 0

    def is_complete(self):
        return np.all(self.pack_col != 0)

    def is_partial(self):
        return np.any(self.pack_col == 0)

    def get_score(self):
        if self.evaluated:
            return self._fitness_value
        else:
            return None

    def set_score(self, value):
        self._fitness_value = value
        self.evaluated = True

    def del_score(self):
        self._fitness_value = None
        self.evaluated = False

    score = property(get_score, set_score, del_score, "fitness value behavior.")

    def get_max_col(self):
        return max(self.pack_col)

    def __lt__(self, val):
        if np.issubdtype(type(val), np.integer):
            return self.pack_col < val
        elif type(val) is PackColSolution and val.evaluated:
            return self.score < val.score

    def __le__(self, val):
        if np.issubdtype(type(val), np.integer):
            return self.pack_col <= val
        elif type(val) is PackColSolution and val.evaluated:
            return self.score <= val.score

    def __eq__(self, val):
        if np.issubdtype(type(val), np.integer):
            return self.pack_col == val
        elif type(val) is PackColSolution and val.evaluated:
            return self.score == val.score

    def __ne__(self, val):
        if np.issubdtype(type(val), np.integer):
            return self.pack_col != val
        elif type(val) is PackColSolution and val.evaluated:
            return self.score != val.score

    def __gt__(self, val):
        if np.issubdtype(type(val), np.integer):
            return self.pack_col > val
        elif type(val) is PackColSolution and val.evaluated:
            return self.score > val.score

    def __ge__(self, val):
        if np.issubdtype(type(val), np.integer):
            return self.pack_col >= val
        elif type(val) is PackColSolution and val.evaluated:
            return self.score >= val.score

    def __len__(self):
        return len(self.pack_col)

    def __getitem__(self, key):
        try:
            return self.pack_col[key]
        except IndexError:
            raise IndexError("index out of bound: {0}".format(key))

    def __setitem__(self, key, value):
        try:
            self.pack_col[key] = value
            if self.evaluated:
                self._fitness_value = None
                self.evaluated = False
        except IndexError:
            raise IndexError("index out of bound: {0}".format(key))
        except ValueError:
            raise ValueError("wrong value type: {0}, {1}".format(
                self.pack_col.dtype, type(value)))

    def __str__(self):
        return str(self.pack_col)
