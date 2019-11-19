import numpy as np
from numpy import linalg as LA
import scipy.spatial.distance as ssd

def timer(start, end):
	hours, rem = divmod(end - start, 3600)
	minutes, seconds = divmod(rem, 60)
	return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def complete_graph(X):
	return ssd.squareform(ssd.pdist(X).astype("float32"))

def lap(A):
	# compute the laplacian of adjacency matrix
	D = np.diag(A.sum(axis=0))
	return D - A

def eigsort(G):
	# Eval, Evec = LA.eig(lap(G))
	if (np.allclose(G, G.T, rtol=1e-05, atol=1e-08)):
		Eval, Evec = LA.eigh(G)
	else:
		print("Input matrix is not symmetric")
		Eval, Evec = LA.eig(G)
		if np.any(Eval.imag > 1e-8) or np.any(Evec.imag > 1e-8):
			print("Complex error")
			exit()
	idx = np.argsort(Eval)
	return Evec[:,idx].astype('float32'), Eval[idx].astype('float32')