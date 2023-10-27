import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

data = np.load("small_world_k5_graphs.npz", allow_pickle=True)
G = data["G"]
A = nx.from_numpy_matrix(G[0, 0][0])
nx.draw(A)
plt.draw()
plt.show()
