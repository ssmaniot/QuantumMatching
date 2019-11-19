from graph_generators import *
import numpy as np

import time
import threading
from misc import timer
from functools import reduce

class ElapsedTimeThread(threading.Thread):
	def __init__(self):
		super(ElapsedTimeThread, self).__init__()
		self._stop_event = threading.Event()

	def stop(self):
		self._stop_event.set()

	def stopped(self):
		return self._stop_event.is_set()

	def run(self):
		thread_start = time.time()
		while not self.stopped():
			print("Elapsed Time {}".format(timer(thread_start, time.time())), end='\r')
			time.sleep(0.01)

"""
TODO: following dataset

	- 10 delaunay
	- 10 k-nn per k=2,3,4,5
	- 10 small-world
	- 10 scale-free
"""

np.random.seed(4513267)

n = 50
G = np.empty((1, 10), dtype='O')
graphs_per_dataset = 10
T = 11
threshold = 0.995

"""
Delaunay graphs
"""
for i in range(graphs_per_dataset):
	G[0,i] = np.empty(T, dtype='O')
	X = np.random.rand(n * 2).reshape((n, 2)).astype('float32')
	g = delaunay_graph(X)
	G[0,i][0] = g

	for t in range(1, T):
		G[0,i][t] = perturbe_graph(g, threshold)
np.savez("delaunay_graphs.npz", G=G)

"""
knn-graphs
"""
# print("knn-graphs\n---------")
K = 5
for k in range(3, K + 1):
	"""
	thread = ElapsedTimeThread()
	thread.start()
	"""
	for i in range(graphs_per_dataset):
		# print("k = {}, {}/{}                              ".format(k, i+1, graphs_per_dataset))
		G[0,i] = np.empty(T, dtype='O')
		X = np.random.rand(n * 2).reshape((n, 2)).astype('float32')
		g = knn_graph(X, k)
		cc, _ = connected_components(g)
		while cc > 1:
			X = np.random.rand(n * 2).reshape((n, 2)).astype('float32')
			g = knn_graph(X, k)
			cc, _ = connected_components(g)
		G[0,i][0] = g

		for t in range(1, T):
			G[0,i][t] = perturbe_graph(g, threshold)
	"""
	thread.stop()
	thread.join()
	"""
	np.savez("knn_k{}_graphs.npz".format(k), G=G)
# print("                                               ", end='\r')

"""
small-world graphs
"""
for i in range(graphs_per_dataset):
	G[0,i] = np.empty(T, dtype='O')
	g = small_world_graph(n)
	G[0,i][0] = g

	for t in range(1, T):
		G[0,i][t] = perturbe_graph(g, threshold)
np.savez("small_world_graphs.npz", G=G)

"""
scale-free graphs
"""
for i in range(graphs_per_dataset):
	G[0,i] = np.empty(T, dtype='O')
	g = scale_free_graph(n)
	cc, _ = connected_components(g)
	while cc > 1:
		g = scale_free_graph(n)
		cc, _ = connected_components(g)
	G[0,i][0] = g

	for t in range(1, T):
		G[0,i][t] = perturbe_graph(g, threshold)
np.savez("scale_free_graphs.npz", G=G)