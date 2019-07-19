import numpy as np
import scipy.spatial.distance as ssd

def mixing_matrix_signature(Evec, Eval, t_min, t_max, dim, threshold):
	# n = G.shape[0]
	# Evec, Eval = np.linal.eig(G)
	# Make sure the eigenvalues are sorted in ascending order
	# idx = np.argsort(Eval)
	# Eval, Evec = Eval[idx], Evec[:,idx]
	n = Evec.shape[0]

	# compute eigenvalue multiplicity
	relative_tol = 1e-6 * np.abs(np.max(Eval))
	unique_eval = [np.min(Eval)]
	a = [0]
	for pos, e in sorted(enumerate(Eval), key=lambda x: x[1]):
		if abs(e - unique_eval[-1]) > relative_tol:
			unique_eval.append(e)
			a.append(pos)
	unique_eval = np.array(unique_eval)

	L = []
	for l in range(0, len(unique_eval) - 1):
		ll = np.arange(a[l], a[l+1])
		L.append(ll)
	L.append(np.arange(a[-1], len(Eval)))

	# compute projectors
	P = np.zeros(len(L) * n ** 2, dtype="float32").reshape((len(L), n, n))
	for l, ll in enumerate(L):
		for k in ll:
			P[l] += np.outer(Evec[:,k], Evec[:,k])

	num_steps = 10
	# signature is diagonal of mixing matrix for increasing t
	MMSdiag = np.zeros(n * num_steps, dtype="float32").reshape((n, num_steps))
	# signature is concatenation of first dim elements of rows for increasing t
	# original code should be "n * num_steps * (dim-1)", reshape((n, num_steps * (dim-1)))
	MMSrow = np.zeros(n * num_steps * (dim - 1), dtype="float32").reshape((n, num_steps * (dim - 1)))

	t_interval = np.logspace(np.log10(t_min), np.log10(t_max), num=num_steps, endpoint=True)

	discarded = 0

	# Precomputing indices to access the upper-triangular section of matrix of pairwise differences between evals
	# WITHOUT the main diagonal (we precompute the sum of "pointwise squared" projectors in PSAME)
	uti = np.triu_indices(len(unique_eval) - 1)
	uti_r, uti_c = uti[0], uti[1] + 1
	# We cast the type to complex64 to prevent casting at each loop to compute np.sinc()
	pairwise_subs = (unique_eval[:, np.newaxis] - unique_eval[np.newaxis, :])[uti_r, uti_c].astype("complex64")
	# The loop is faster if we cast this sum to complex64 instead of casting P to complex64 at the moment of its creation
	PSAME = np.sum(P * P, axis=0).astype("complex64")
	for j, t in enumerate(t_interval):
		"""
		The computation of PSINC and M is the most expensive one in the loop.
		"""
		mask = t * np.abs(pairwise_subs) / np.pi <= threshold
		PSINC = P[uti_r[mask]] * P[uti_c[mask]]
		SINC = 2.0 * np.sinc(t * pairwise_subs[mask] / np.pi)
		# same result/performance of np.tensordot()
		# M = PSAME + np.einsum('ijk,i->jk', PSINC, SINC, casting="same_kind", optimize=True)
		M = PSAME + np.tensordot(PSINC, SINC, ([0,0]))
		discarded += np.count_nonzero(mask)

		M = np.real(M)

		MMSdiag[:,j] = np.diag(M)
		Q = np.quantile(M, np.arange(1/dim, 1-1e-8, 1/dim), axis=1, interpolation="midpoint")
		M = np.fliplr(Q.T)
		MMSrow[:, np.arange(j*(dim-1), (j+1)*(dim-1))] = M[:, np.arange(dim-1)]

	num_of_discarded_pairs = discarded / (len(t_interval) * len(unique_eval) ** 2)

	return MMSdiag, MMSrow, num_of_discarded_pairs

def heat_kernel_signature(Evec, Eval, dim):
	# n = G.shape[0]
	# Evec, Eval = np.linal.eig(G)
	# Make sure the eigenvalues are sorted in ascending order
	# idx = np.argsort(Eval)
	# Eval, Evec = Eval[idx], Evec[:,idx]
	n = Evec.shape[0]
	
	# compute eigenvalue multiplicity
	relative_tol = 1e-8 * abs(max(Eval))
	unique_eval = [min(Eval)]
	a = [0]
	for pos, e in sorted(enumerate(Eval), key=lambda x: x[1]):
		if abs(e - unique_eval[-1]) > relative_tol:
			unique_eval.append(e)
			a.append(pos)
	unique_eval = np.array(unique_eval)

	L = []
	for l in range(0, len(unique_eval) - 1):
		ll = np.arange(a[l], a[l+1])
		L.append(ll)
	L.append(np.arange(a[-1], len(Eval)))

	# compute projectors
	P = np.zeros(len(L) * n ** 2).reshape((len(L), n, n))
	for l, ll in enumerate(L):
		for k in ll:
			P[l] += np.outer(Evec[:,k], Evec[:,k])

	num_steps = 100
	# signature is diagonal of mixing matrix for increasing t
	HKSdiag = np.zeros(n * num_steps).reshape((n, num_steps))
	# signature is concatenation of first dim elements of rows for increasing t
	# original code should be "n * num_steps * (dim-1)", reshape((n, num_steps * (dim-1)))
	HKSrow = np.zeros(n * num_steps * (dim - 1)).reshape((n, num_steps * (dim - 1)))

	tmin = abs(4.0 * np.log(10.0) / Eval[-1])
	tmax = abs(4.0 * np.log(10.0) / Eval[1])

	# t_interval = np.arange(tmin, tmax, (tmax - tmin) / num_steps)
	# stepsize = (np.log(tmax) - np.log(tmin)) / num_steps
	# logts = np.arange(np.log(tmin), np.log(tmax), stepsize)
	t_interval = np.exp(np.linspace(np.log(tmin), np.log(tmax), num=num_steps, endpoint=True))

	for i, t in enumerate(t_interval):
		H = np.einsum('ijk,i->jk', P, np.exp(-unique_eval * t)) #np.zeros(n ** 2).reshape((n, n))
		#for j, l in enumerate(unique_eval):
		#	H += P[j] * np.exp(-l * t)
		HKSdiag[:,i] = H.diagonal()
		Q = np.quantile(H, np.arange(1/dim, 1-1e-8, 1/dim), axis=1, interpolation="midpoint")
		H = np.fliplr(Q.T) #(np.sort(Q.T, axis=1), axis=1)
		HKSrow[:, np.arange(i*(dim-1), (i+1)*(dim-1))] = H[:, np.arange(dim-1)]

	return HKSdiag, HKSrow

def wave_kernel_signature(Evec, Eval):
	# n = G.shape[0]
	# Evec, Eval = np.linal.eig(G)
	# Make sure the eigenvalues are sorted in ascending order
	# idx = np.argsort(Eval)
	# Eval, Evec = Eval[idx], Evec[:,idx]
	n = Evec.shape[0]
	
	E = abs(np.real(Eval))
	PHI = np.real(Evec)
	
	wks_variance = 6
	N = 100
	WKS = np.zeros(n*N).reshape((n, N))
	
	log_E = np.log(np.maximum(abs(E), 1e-6 * np.ones(n)))
	e = np.linspace(log_E[1], max(log_E)/1.02, N)
	sigma = (e[1] - e[0]) * wks_variance
	
	C = np.zeros(N) # weights used for the normalization of f_E
	
	# Could be vectorized
	for i in range(N):
		WKS[:,i] = np.sum(np.multiply(np.power(PHI, 2), 
			np.tile(np.exp(-np.power((e[i] - log_E), 2) / (2 * sigma ** 2)), (n, 1))), axis=1)
		C[i] = np.sum(np.exp(-np.power((e[i] - log_E), 2) / (2 * sigma ** 2)))
	
	# normalize WKS 
	WKS = WKS / np.tile(C, (n, 1))
	return WKS