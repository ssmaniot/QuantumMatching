import numpy as np

postfix = ["_dmin", "_d10", "_d5"]
prefix = ["car", "houses_full_reduced", "moto", "socnet"]
knn = ["knn_k{}_graphs".format(i) for i in range(3, 6)]
small_world = ["small_world_k{}_graphs".format(i) for i in range(4, 6)]
datasets = ["delaunay_graphs", "scale_free_graphs"] + knn + small_world + [pre + post for post in postfix for pre in prefix]
results = ["result_{}.npz".format(dataset) for dataset in datasets]

data = np.load("MAT/result_car_dmin.npy")
# print(datasets)
# data = np.load("MAT/{}".format(results[0]))
print(data.files)
outcomeMMS = data['outcomeMMS']
print(outcomeMMS[:,:,:,2])
exit()
print('mean: {}'.format(np.mean(outcomeMMS[0][ ::2])))
print('  sd: {}'.format(np.mean(outcomeMMS[0][1::2])))
