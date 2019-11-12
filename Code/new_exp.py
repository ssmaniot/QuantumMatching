from signatures import *
from misc import *
from matching import compute_matching
import numpy as np
import scipy.io
import os
import time

np.random.seed(46751)

start = time.time()

dataset = "houses_full_reduced" # "socnet"
path = "MAT/{}.mat".format(dataset)

data = scipy.io.loadmat(path)
G = data["G"]
n = G.shape[1]
quantiles = min(30, min([G[0,i].shape[0] for i in range(n)]))

t_min = 1e-1
t_max = np.linspace(1, 20, num=8, endpoint=True) # go to 100
quantiles = np.arange(5, quantiles + 1, step=5)
thresholds = [np.Inf] # np.logspace(-8, 9, num=8)# [np.Inf]

"""
HKS: Evec, Eval, dim
WKS: Evec, Eval
MMS: Evec, Eval, t_min, t_max, dim, threshold

where 'dim' are the quantiles

=> outcomeHW[quantile][HKS, errHKS, WKS, errWKS]
=> outcomeMMS[quantile][t_max][threshold][MMSdiag, errMMSdiag, MMSrow, errMMSrow]
"""

# INIZIO ESPERIMENTI

outcomesHW = np.empty((len(quantiles), 4))
outcomesMMS = np.empty((len(quantiles), len(t_max), len(thresholds), 4))
num_pairs = n * (n - 1) // 2
tot = num_pairs  * len(quantiles) * len(thresholds) * len(t_max)
i = 0

avg_discarded = 0.0
pair = 0
resultHW = np.empty((num_pairs, len(quantiles), 2))
resultMMS = np.empty((num_pairs, len(quantiles), len(t_max), len(thresholds), 2))

for g1 in range(n):
	G1 = G[0, g1]
	n1 = G1.shape[0]

	for g2 in range(g1 + 1, n):
		G2 = G[0, g2]
		n2 = G2.shape[0]
		
		min_dim = min(n1, n2)
		ground_truth = np.random.permutation(min_dim)
		P = np.zeros(n2 ** 2).reshape((n2, n2))
		for r in range(n2):
			if r <= min_dim:
				P[r, ground_truth[r]] = 1
			else:
				P[r, r] = 1
		G2 = P.T @ G2 @ P

		quantile_resultHW = np.empty((0, 2))
		quantile_resultMMS = np.empty((len(quantiles), len(t_max), len(thresholds), 2))

		for di, d in enumerate(quantiles):
			PHI1, E1 = eigsort(lap(G1))
			HKSdiag1, _ = heat_kernel_signature(PHI1, E1, d)
			WKSdiag1 = wave_kernel_signature(PHI1, E1)

			PHI2, E2 = eigsort(lap(G2))
			HKSdiag2, _ = heat_kernel_signature(PHI2, E2, d)
			WKSdiag2 = wave_kernel_signature(PHI2, E2)

			# HKS
			assignment_HKSdiag, num_matches_hks_diag = compute_matching(HKSdiag1, HKSdiag2, ground_truth)
			# assignment_HKSrow, num_matches_hks_row = compute_matching(HKSrow1, HKSrow2, ground_truth)
			
			# WKS
			assignment_WKSdiag, num_matches_wks_diag = compute_matching(WKSdiag1, WKSdiag2, ground_truth)

			quantile_resultHW = np.vstack([quantile_resultHW, np.array([
				num_matches_hks_diag / min_dim, # num_matches_hks_row / min_dim,
				num_matches_wks_diag / min_dim])])

			PHI1, E1 = eigsort(G1)
			PHI2, E2 = eigsort(G2)

			for tmi, tm in enumerate(t_max):
				for ti, threshold in enumerate(thresholds):
					print("{:.3f}%".format(i / tot * 100), end="\r")
					i += 1

					MMSdiag1, MMSrow1, num_discarded = mixing_matrix_signature(PHI1, E1, t_min, tm, d, threshold)
					MMSdiag2, MMSrow2, _ = mixing_matrix_signature(PHI2, E2, t_min, tm, d, threshold)

					# MMS
					assignment_MMSdiag, num_matches_mms_diag = compute_matching(MMSdiag1, MMSdiag2, ground_truth)
					assignment_MMSrow, num_matches_mms_row = compute_matching(MMSrow1, MMSrow2, ground_truth)

					quantile_resultMMS[di, tmi, ti] = num_matches_mms_diag / min_dim, num_matches_mms_row / min_dim

		resultHW[pair] = quantile_resultHW
		resultMMS[pair] = quantile_resultMMS
		pair += 1

mean_accuracyHW = np.mean(resultHW, axis=0)
stderr_accuracyHW = np.std(resultHW, axis=0) / np.sqrt(resultHW.shape[0])
outcomesHW[..., ::2] = mean_accuracyHW
outcomesHW[..., 1::2] = stderr_accuracyHW

mean_accuracyMMS = np.mean(resultMMS, axis=0)
stderr_accuracyMMS = np.std(resultMMS, axis=0) / np.sqrt(resultMMS.shape[0])
outcomesMMS[..., ::2] = mean_accuracyMMS
outcomesMMS[..., 1::2] = stderr_accuracyMMS

# FINE ESPERIMENTI

print("{:.3f}%".format(i / tot * 100))
end = time.time()

np.savez("MAT/result_{}_thr.npz".format(dataset), outcomeHW = outcomesHW, outcomeMMS = outcomesMMS,
	quantiles = quantiles, t_max = t_max, thresholds = thresholds)

timer(start, end)