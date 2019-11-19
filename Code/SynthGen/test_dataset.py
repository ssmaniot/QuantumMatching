import os
import numpy as np 
from scipy.sparse.csgraph import connected_components

"""
Verify that all graphs in all datasets are symmetric and connected.
We need to check the 1st graph of each sample group because all the
others are permutations of its adjacency matrix.
"""

datasets = [f for f in os.listdir('.') if os.path.isfile(f) and len(f) > 4 and f[-4:] == ".npz"]
for dataset in datasets:
	data = np.load(dataset, allow_pickle=True)
	G = data["G"]
	valid = True
	for i in range(G.shape[1]):
		g = G[0,i][0]
		cc, _ = connected_components(g)
		if cc > 1 or not ((g == g.T).all()):
			valid = False
			break
	if valid:
		print("[ OK ] {}".format(dataset))
	else:
		print("[FAIL] {}".format(dataset))