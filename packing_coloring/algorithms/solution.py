# -*- coding: utf-8 -*-

import numpy as np
from packing_coloring.algorithms.problem import *

class PackColSolution:
	def __init__(self, g_prob):
		self.pack_col = np.zeros(g_prob.v_size, dtype=int)
		self._fitness_value = None
		self.evaluated = False

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

	def __lt__(self, sol):
		pass

	def __le__(self, sol):
		pass

	def __eq__(self, sol):
		pass

	def __ne__(self, sol):
		pass

	def __gt__(self, sol):
		pass

	def __ge__(self, sol):
		pass

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
			raise ValueError("wrong value type: {0}, {1}".format(self.pack_col.dtype, type(value)))

	def __str__(self):
		return str(self.pack_col)