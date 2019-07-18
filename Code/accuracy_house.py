from signatures import *
from misc import *
from matching import compute_matching
import numpy as np
import scipy.io
import os
import time

start = time.time()

path = "MAT/houses_full_reduced.mat"

data = scipy.io.loadmat(path)
G = data["G"]
n = G.shape[1]
t_min = 1e-1
t_max = 1

for threshold in np.linspace(0.1, 1, num=10, endpoint=True): #[np.Inf]:
	avg_discarded = 0.0
	result = np.empty((0, 5))
	print("threshold = {}".format(threshold))

	for g1 in range(n):
		print("Cycle {}/{}".format(g1 + 1, n))
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

			# use first d quantiles
			d = min_dim

			PHI1, E1 = eigsort(lap(G1))
			HKSdiag1, HKSrow1 = heat_kernel_signature(PHI1, E1, d)
			WKSdiag1 = wave_kernel_signature(PHI1, E1)

			PHI2, E2 = eigsort(lap(G2))
			HKSdiag2, HKSrow2 = heat_kernel_signature(PHI2, E2, d)
			WKSdiag2 = wave_kernel_signature(PHI2, E2)

			PHI1, E1 = eigsort(G1)
			MMSdiag1, MMSrow1, num_discarded = mixing_matrix_signature(PHI1, E1, t_min, t_max, d, threshold)

			PHI2, E2 = eigsort(G2)
			MMSdiag2, MMSrow2, _ = mixing_matrix_signature(PHI2, E2, t_min, t_max, d, threshold)

			# HKS
			assignment_HKSdiag, num_matches_hks_diag = compute_matching(HKSdiag1, HKSdiag2, ground_truth)
			assignment_HKSrow, num_matches_hks_row = compute_matching(HKSrow1, HKSrow2, ground_truth)
			
			# WKS
			assignment_WKSdiag, num_matches_wks_diag = compute_matching(WKSdiag1, WKSdiag2, ground_truth)

			# MMS
			assignment_MMSdiag, num_matches_mms_diag = compute_matching(MMSdiag1, MMSdiag2, ground_truth)
			assignment_MMSrow, num_matches_mms_row = compute_matching(MMSrow1, MMSrow2, ground_truth)

			result = np.vstack([result, np.array([
				num_matches_hks_diag / min_dim, num_matches_hks_row / min_dim,
				num_matches_wks_diag / min_dim,
				num_matches_mms_diag / min_dim, num_matches_mms_row / min_dim])])

	mean_accuracy = np.mean(result, axis=0)
	stderr_accuracy = np.std(result, axis=0) / np.sqrt(result.shape[0])
	print("mean accuracy:\n{}".format(mean_accuracy))
	print("stderr accuracy:\n{}".format(stderr_accuracy))

end = time.time()
print("Time elapsed = {:.3f}s".format(end - start))