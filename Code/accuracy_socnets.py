from signatures import *
from misc import *
from matching import compute_matching
import numpy as np
import scipy.io
import os
import time

np.random.seed(46751)

start = time.time()

dataset = "socnet"
path = "MAT/{}.mat".format(dataset)
experiments = 1

data = scipy.io.loadmat(path)
G = data["G"]
n = G.shape[1]
t_min = 1e-1
t_max = 1
threshold = np.Inf

outcomes = np.zeros(experiments * 8, dtype="float64").reshape((experiments, 8))
idx = np.arange(4)
round_per_exp = n * (n-1) / 2 + n
tot = round_per_exp * experiments
i = 0

for experiment in range(experiments):
	result = np.empty((0, 4))
	
	for g1 in range(n):
		G1 = G[0, g1].astype("float32")
		n1 = G1.shape[0]

		for g2 in range(g1, n):
			G2 = G[0, g2].astype("float32")
			n2 = G2.shape[0]

			print("{:.3f}%".format(i / tot * 100), end="\r")
			i += 1
			
			min_dim = min(n1, n2)
			ground_truth = np.random.permutation(min_dim)
			P = np.zeros(n2 ** 2, dtype="float32").reshape((n2, n2))
			for r in range(n2):
				if r <= min_dim:
					P[r, ground_truth[r]] = 1.0
				else:
					P[r, r] = 1.0	
			G2 = P.T @ G2 @ P

			# use first d quantiles
			d = 10#min_dim

			PHI1, E1 = eigsort(lap(G1))
			HKSdiag1, _ = heat_kernel_signature(PHI1, E1, d)
			WKSdiag1 = wave_kernel_signature(PHI1, E1)

			PHI2, E2 = eigsort(lap(G2))
			HKSdiag2, _ = heat_kernel_signature(PHI2, E2, d)
			WKSdiag2 = wave_kernel_signature(PHI2, E2)

			PHI1, E1 = eigsort(G1)
			qstart = time.time()
			MMSdiag1, MMSrow1, num_discarded = mixing_matrix_signature(PHI1, E1, t_min, t_max, d, threshold)

			PHI2, E2 = eigsort(G2)
			MMSdiag2, MMSrow2, _ = mixing_matrix_signature(PHI2, E2, t_min, t_max, d, threshold)

			# HKS
			assignment_HKSdiag, num_matches_hks_diag = compute_matching(HKSdiag1, HKSdiag2, ground_truth)
			# assignment_HKSrow, num_matches_hks_row = compute_matching(HKSrow1, HKSrow2, ground_truth)
			
			# WKS
			assignment_WKSdiag, num_matches_wks_diag = compute_matching(WKSdiag1, WKSdiag2, ground_truth)

			# MMS
			assignment_MMSdiag, num_matches_mms_diag = compute_matching(MMSdiag1, MMSdiag2, ground_truth)
			assignment_MMSrow, num_matches_mms_row = compute_matching(MMSrow1, MMSrow2, ground_truth)

			result = np.vstack([result, np.array([
				num_matches_hks_diag / min_dim, # num_matches_hks_row / min_dim,
				num_matches_wks_diag / min_dim,
				num_matches_mms_diag / min_dim, num_matches_mms_row / min_dim])])

	mean_accuracy = np.mean(result, axis=0)
	stderr_accuracy = np.std(result, axis=0) / np.sqrt(result.shape[0])
	outcomes[experiment][idx * 2] = mean_accuracy
	outcomes[experiment][idx * 2 + 1] = stderr_accuracy

print("{:.3f}%".format(i / tot * 100))
end = time.time()

np.save("MAT/result_{}_d10.npy".format(dataset), outcomes)

print("Time elapsed = {:.3f}s".format(end - start))