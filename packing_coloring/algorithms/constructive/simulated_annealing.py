# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
import random as rd
import math

class Problem:
    def __init__(self, size, dist_m):
        self.dist_matrix = dist_m
        self.size_n = size

    def non_coloured(self, pattern_k):
        return np.sum(pattern_k == 0)

    def free_for_colour(self, color_c, pattern_k):
        return np.sum(self.give_k_colorable_set(pattern_k, color_c))

    def give_k_colorable_set(self, pattern_k, k_col):
        k_coloried = (pattern_k == k_col)
        le_k_dist = (self.dist_matrix <= k_col)
        no_k_dist = np.logical_not(np.any(np.logical_and(k_coloried, le_k_dist), axis=1))
        k_colorable_set = np.logical_and(no_k_dist, pattern_k==0)

        return k_colorable_set

    def plant_colour(self, pattern_k, color_c, temp_t, k):
        pattern_l = np.copy(pattern_k)

        while self.free_for_colour(color_c, pattern_l) > 0:
            v = None
            f_v = float("infinity")

            for i in range(k):
                choosable = np.arange(len(pattern_l))[self.give_k_colorable_set(pattern_l, color_c)]
                w = choosable[np.random.randint(len(choosable))]
                w_colored = np.copy(pattern_l)
                w_colored[w] = color_c
                f_w = self.free_for_colour(color_c, pattern_l) - self.free_for_colour(color_c, w_colored)

                if rd.random() < sp.N(sp.exp((f_v - f_w)/temp_t)):
                    f_v = f_w
                    v = w

            if v != None:
                pattern_l[v] = color_c

        return pattern_l


    def solve(self, t_max, t_min, q, k):
        best_patterns = np.array([]).reshape(0, self.size_n)
        pattern_k = np.zeros(self.size_n, dtype=int)
        best_patterns = np.append(best_patterns, pattern_k.reshape(1, self.size_n), axis=0)
        color_c = 1

        while len(best_patterns) == 0 or self.non_coloured(best_patterns[0]) > 0:
            new_patterns = np.array([]).reshape(0, self.size_n)
            p = self.non_coloured(pattern_k)
            m = 0

            for pattern in best_patterns:
                temp = t_max
                while temp > t_min:
                    pattern_t = self.plant_colour(pattern, color_c, temp, k)
                    a = p - self.non_coloured(pattern_t)

                    if a >= m:
                        if a > m:
                            new_patterns = np.array([]).reshape(0,self.size_n)
                            m = a
                        if not np.any(np.all(pattern_t == new_patterns, axis=1)):
                            new_patterns = np.append(new_patterns, pattern_t.reshape(1, self.size_n), axis=0)


                    temp = temp * q
            best_patterns = new_patterns

            color_c += 1
        return best_patterns[0]


