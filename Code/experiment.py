from signatures import *
from misc import *
from matching import compute_matching
import numpy as np
import scipy.io
import os
import time

np.random.seed(46751)

start = time.time()

dataset = "houses_full_reduced"
path = "MAT/{}.mat".format(dataset)
experiments = 10

data = scipy.io.loadmat(path)
G = data["G"]
n = G.shape[1]
t_min = 1e-1
t_max = 1
quantiles = 10
thresholds = np.logspace(np.log10(1e-8), np.log10(100), num=10, endpoint=True) # [np.Inf]

min_dim = np.Inf
# determine minimum graph size
for g in range(n):
	s = G[0, g].shape[0]
	if s < min_dim:
		min_dim = s

quantiles = min(s, quantiles)

outcomesHW = np.empty((experiments, quantiles, 4))
outcomesMMS = np.empty((experiments, len(thresholds), quantiles, 4))
idx = np.arange(4)
round_per_exp = n * (n-1) // 2
tot = round_per_exp * experiments * quantiles * len(thresholds)
i = 0

# HKS/WKS: outcomesHW[experiment][quantile][HKS, errHKS, WKS, errWKS]
# MMS: outcomesMMS[experiment][threshold][quantile][MMSdiag, errMMSdiag, MMSrow, errMMSrow]

for experiment in range(experiments):
	avg_discarded = 0.0
	pair = 0
	resultHW = np.empty((round_per_exp, quantiles, 2))
	resultMMS = np.empty((round_per_exp, len(thresholds), quantiles, 2))

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
			quantile_resultMMS = np.empty((len(thresholds), quantiles, 2))
			#for di, d in enumerate(np.arange(min_dim - quantiles + 1, min_dim + 1)):
			for di, d in enumerate(np.arange(1, quantiles + 1)):
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

				threshold_resultMMS = np.empty((0, 2))

				for ti, threshold in enumerate(thresholds):
					print("{:.3f}%".format(i / tot * 100), end="\r")
					i += 1
					
					MMSdiag1, MMSrow1, num_discarded = mixing_matrix_signature(PHI1, E1, t_min, t_max, d, threshold)
					MMSdiag2, MMSrow2, _ = mixing_matrix_signature(PHI2, E2, t_min, t_max, d, threshold)

					# MMS
					assignment_MMSdiag, num_matches_mms_diag = compute_matching(MMSdiag1, MMSdiag2, ground_truth)
					assignment_MMSrow, num_matches_mms_row = compute_matching(MMSrow1, MMSrow2, ground_truth)

					threshold_resultMMS = np.vstack([threshold_resultMMS, np.array([
						num_matches_mms_diag / min_dim, num_matches_mms_row / min_dim])])

				quantile_resultMMS[:, di] = threshold_resultMMS

			resultHW[pair] = quantile_resultHW
			resultMMS[pair] = quantile_resultMMS
			pair += 1

	mean_accuracyHW = np.mean(resultHW, axis=0)
	stderr_accuracyHW = np.std(resultHW, axis=0) / np.sqrt(resultHW.shape[0])
	outcomesHW[experiment, ..., ::2] = mean_accuracyHW
	outcomesHW[experiment, ..., 1::2] = stderr_accuracyHW

	mean_accuracyMMS = np.mean(resultMMS, axis=0)
	stderr_accuracyMMS = np.std(resultMMS, axis=0) / np.sqrt(resultMMS.shape[0])
	outcomesMMS[experiment, ..., ::2] = mean_accuracyMMS
	outcomesMMS[experiment, ..., 1::2] = stderr_accuracyMMS

print("{:.3f}%".format(i / tot * 100))
end = time.time()

np.savez("MAT/result_{}_thr.npz".format(dataset), outcomeHW = outcomesHW, outcomeMMS = outcomesMMS,
	thresholds = thresholds, quantiles = np.arange(min_dim - quantiles + 1, min_dim + 1))

timer(start, end)