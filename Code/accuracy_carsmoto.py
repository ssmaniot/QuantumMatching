from kernel import *
from misc import *
from matching import compute_matching
import numpy as np
import scipy.io
import os
import time

start = time.time()

dataset = "car"
path = "MAT/car_moto_pairs/{}_pairs/".format(dataset)

if dataset == "car":
	num_pairs = 30
	t_min = 1e-2
else:
	num_pairs = 20
	t_min = 1e-3

t_max = 1
result = np.empty((0, 5))

for i in range(1, num_pairs + 1):
	fname = "pair_{}.mat".format(i)
	data = scipy.io.loadmat(path + fname)
	print("Cycle {}, {}".format(i, fname))
	
	n = data["nF1"][0,0]
	G1 = complete_graph(data["features1"][np.arange(n), 0:2])
	G2 = complete_graph(data["features2"][np.arange(n), 0:2])

	ground_truth = np.random.permutation(n)
	P = np.zeros(n ** 2).reshape((n, n))
	for r in range(n):
		P[r, ground_truth[r]] = 1
	G2 = P.T @ G2 @ P

	# use first d quantiles
	d = n
	threshold = np.Inf

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
		num_matches_hks_diag / n, num_matches_hks_row / n,
		num_matches_wks_diag / n,
		num_matches_mms_diag / n, num_matches_mms_row / n])])

mean_accuracy = np.mean(result, axis=0)
stderr_accuracy = np.std(result, axis=0) / np.sqrt(result.shape[0])

end = time.time()

print("Time elapsed = {:.3f}s".format(end - start))
print("mean accuracy:\n{}".format(mean_accuracy))
print("stderr accuracy:\n{}".format(stderr_accuracy))