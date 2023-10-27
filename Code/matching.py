import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_matching(S1, S2, ground_truth):
    dist = cdist(S1, S2)
    _, col_ind = linear_sum_assignment(dist)
    # assignment = np.zeros(ground_truth.shape[0] ** 2).reshape(ground_truth.shape)
    # assignment[row_ind, col_ind] = 1
    # print(ground_truth[ground_truth == col_ind])
    return col_ind, (ground_truth == col_ind).sum()
