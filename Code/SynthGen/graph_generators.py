import numpy as np
from scipy.spatial import Delaunay
import scipy.spatial.distance as ssd
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
import networkx as nx


def small_world_graph(n, k=2, p=0.1):
    G = nx.connected_watts_strogatz_graph(n, k, p)
    while nx.algorithms.smallworld.sigma(G, niter=10) <= 1.0:
        G = nx.connected_watts_strogatz_graph(n, k, p)
    return nx.adjacency_matrix(G).todense().astype("float32")


def scale_free_graph(n):
    return (
        nx.adjacency_matrix(nx.Graph(nx.scale_free_graph(n)))
        .todense()
        .astype("float32")
    )


def delaunay_graph(X):
    n = X.shape[0]
    tri = Delaunay(X, qhull_options="QJ Pp").simplices

    G = np.zeros(n**2, dtype="float32").reshape((n, n))
    for i in range(tri.shape[0]):
        G[tri[i, 0], tri[i, 1]] = 1.0
        G[tri[i, 1], tri[i, 2]] = 1.0
        G[tri[i, 0], tri[i, 2]] = 1.0
    return (G + G.T > 0.0).astype("float32")


def knn_graph(X, k):
    n = X.shape[0]
    D = ssd.squareform(ssd.pdist(X).astype("float32"))
    idx = np.argsort(D, axis=1)
    G = np.zeros(n**2, dtype="float32").reshape((n, n))
    for i in range(n):
        for j in range(k):
            G[i, idx[i, j]] = 1.0
            G[idx[i, j], i] = 1.0
    return G


def perturbe_graph(G, threshold=0.995):
    n = G.shape[0]
    P = np.random.rand(n**2).reshape((n, n)).astype("float32")
    P = np.triu(P) + np.triu(P).T
    g = np.abs(G - (P > threshold).astype("float32"))
    cc, _ = connected_components(g)
    while cc > 1:
        P = np.random.rand(n**2).reshape((n, n)).astype("float32")
        P = np.triu(P) + np.triu(P).T
        g = np.abs(G - (P > threshold).astype("float32"))
        cc, _ = connected_components(g)
    return g
