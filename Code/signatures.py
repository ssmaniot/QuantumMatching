import numpy as np

def compute_eigenvalue_multiplicity(Eval, tol):
	relative_tol = tol * np.abs(np.max(Eval))
	unique_eval = [np.min(Eval)]
	a = [0]
	for pos, e in sorted(enumerate(Eval), key=lambda x: x[1]):
		if abs(e - unique_eval[-1]) > relative_tol:
			unique_eval.append(e)
			a.append(pos)
	unique_eval = np.array(unique_eval)
	return unique_eval, a

def compute_projectors(Eval, Evec, unique_eval, a, n):
	L = []
	for l in range(0, len(unique_eval) - 1):
		ll = np.arange(a[l], a[l+1])
		L.append(ll)
	L.append(np.arange(a[-1], len(Eval)))

	# Compute projectors
	P = np.zeros(len(L) * n ** 2).reshape((len(L), n, n))
	for l, ll in enumerate(L):
		for k in ll:
			P[l] += np.outer(Evec[:,k], Evec[:,k])
	return P

def MMS_row(n, num_steps, dim):
	return np.empty(n * num_steps * dim*(dim-1)//2, dtype="float32").reshape((n, num_steps * dim*(dim-1)//2))

def quantile(M, dim):
	return np.vstack([np.quantile(M, np.arange(1/q, 1-1e-8, 1/q), axis=1, interpolation="midpoint") for q in np.arange(2, dim + 1)]).T

def mixing_matrix_signature(Evec, Eval, t_min, t_max, dim, threshold):
	n = Evec.shape[0]

	# Compute eigenvalue multiplicity
	unique_eval, a = compute_eigenvalue_multiplicity(Eval, 1e-6)

	# Compute projectors
	P = compute_projectors(Eval, Evec, unique_eval, a, n)

	num_steps = 10
	# Signature is diagonal of mixing matrix for increasing t
	MMSdiag = np.zeros(n * num_steps, dtype="float32").reshape((n, num_steps))
	
	# Signature is concatenation of first dim elements of rows for increasing t
	# original code should be "n * num_steps * (dim-1)", reshape((n, num_steps * (dim-1)))
	MMSrow = MMS_row(n, num_steps, dim)

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
		# The computation of PSINC and M is the most expensive one in the loop.
		mask = t * np.abs(pairwise_subs) / np.pi <= threshold
		PSINC = P[uti_r[mask]] * P[uti_c[mask]]
		SINC = 2.0 * np.sinc(t * pairwise_subs[mask] / np.pi)

		# same result/performance of np.tensordot()
		# M = PSAME + np.einsum('ijk,i->jk', PSINC, SINC, casting="same_kind", optimize=True)
		M = PSAME + np.tensordot(PSINC, SINC, ([0,0]))
		discarded += np.count_nonzero(mask)
		M = np.real(M)
		MMSdiag[:,j] = np.diag(M)
		"""
		OLD CODE
		Q = np.quantile(M, np.arange(1/dim, 1-1e-8, 1/dim), axis=1, interpolation="midpoint")
		M = np.fliplr(np.sort(Q.T, axis=1))
		M = old_quantile(M, dim)
		MMSrow[:, np.arange(j*(dim-1), (j+1)*(dim-1))] = M[:, np.arange(dim-1)]
		"""
		MMSrow[:, np.arange(j*dim*(dim-1)//2, (j+1)*dim*(dim-1)//2)] = quantile(M, dim)

	num_of_discarded_pairs = discarded / (len(t_interval) * len(unique_eval) ** 2)
	return MMSdiag, MMSrow, num_of_discarded_pairs

def heat_kernel_signature(Evec, Eval, dim):
	n = Evec.shape[0]
	
	# Compute eigenvalue multiplicity
	unique_eval, a = compute_eigenvalue_multiplicity(Eval, 1e-8)

	# Compute projectors
	P = compute_projectors(Eval, Evec, unique_eval, a, n)

	num_steps = 100
	
	# Signature is diagonal of mixing matrix for increasing t
	HKSdiag = np.zeros(n * num_steps).reshape((n, num_steps))

	# Signature is concatenation of first dim elements of rows for increasing t
	# original code should be "n * num_steps * (dim-1)", reshape((n, num_steps * (dim-1)))
	HKSrow = np.zeros(n * num_steps * (dim - 1)).reshape((n, num_steps * (dim - 1)))

	tmin = abs(4.0 * np.log(10.0) / Eval[-1])
	tmax = abs(4.0 * np.log(10.0) / Eval[1])

	t_interval = np.exp(np.linspace(np.log(tmin), np.log(tmax), num=num_steps, endpoint=True))

	for i, t in enumerate(t_interval):
		H = np.einsum('ijk,i->jk', P, np.exp(-unique_eval * t))
		# np.zeros(n ** 2).reshape((n, n))
		# for j, l in enumerate(unique_eval):
		# 	H += P[j] * np.exp(-l * t)
		HKSdiag[:,i] = H.diagonal()
		Q = np.quantile(H, np.arange(1/dim, 1-1e-8, 1/dim), axis=1, interpolation="midpoint")
		H = np.fliplr(Q.T) 
		# (np.sort(Q.T, axis=1), axis=1)
		HKSrow[:, np.arange(i*(dim-1), (i+1)*(dim-1))] = H[:, np.arange(dim-1)]

	return HKSdiag, HKSrow

def wave_kernel_signature(Evec, Eval):
	n = Evec.shape[0]
	
	E = abs(np.real(Eval))
	PHI = np.real(Evec)
	
	wks_variance = 6
	N = 100
	WKS = np.zeros(n*N).reshape((n, N))
	
	log_E = np.log(np.maximum(abs(E), 1e-6 * np.ones(n)))
	e = np.linspace(log_E[1], max(log_E)/1.02, N)
	sigma = (e[1] - e[0]) * wks_variance
	
	# Weights used for the normalization of f_E
	C = np.zeros(N) 
	
	# Could be vectorized
	for i in range(N):
		WKS[:,i] = np.sum(np.multiply(np.power(PHI, 2), 
			np.tile(np.exp(-np.power((e[i] - log_E), 2) / (2 * sigma ** 2)), (n, 1))), axis=1).ravel()
		C[i] = np.sum(np.exp(-np.power((e[i] - log_E), 2) / (2 * sigma ** 2)))
	
	# normalize WKS 
	WKS = WKS / np.tile(C, (n, 1))
	return WKS