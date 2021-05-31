import numpy as np

real = ["car", "houses_full_reduced", "moto", "socnet"]
synthetic = ["delaunay_graphs", "knn_k3_graphs", "knn_k4_graphs", "knn_k5_graphs", "scale_free_graphs", "small_world_k4_graphs", "small_world_k5_graphs"]

print('')
print('Real Datasets')
print('~~~~~~~~~~~~~')
for dataset in real:
	dmin = np.load('MAT/result_{}_dmin.npy'.format(dataset))
	d5   = np.load('MAT/result_{}_d5.npy'.format(dataset))
	d10  = np.load('MAT/result_{}_d10.npy'.format(dataset))
	data = np.hstack([dmin[:,:4], d5[:,:2], d10[:,:2], dmin[:,4:], d5[:,4:], d10[:,4:]])
	m = np.mean(data[:,0::2], axis = 0)
	sd = np.mean(data[:,1::2], axis = 0)
	print(dataset)
	print('-' * len(dataset))
	print('       HKSmin WKS    HKSd5  HKSd10 MMRmin MMDmin MMRd5  MMDd5  MMRd10 MMDd10')
	print('mean: {}'.format(np.round(m, 4)))
	print('  sd: {}'.format(np.round(sd, 4)))
	print('\n')
	
print('Synthetic Datasets')
print('~~~~~~~~~~~~~~~~~~')
for dataset in synthetic:
	data = np.load('MAT/result_{}.npz'.format(dataset))
	hw = data['outcomeHW']
	mms = data['outcomeMMS']
	qs = data['quantiles'].shape[0]
	m =  np.hstack([hw[0,0], hw[0,2]] + [hw[q,0] for q in range(1, qs)] + [mms[q,0,0,::2] for q in range(qs)])
	sd =  np.hstack([hw[0,1], hw[0,3]] + [hw[q,1] for q in range(1, qs)] + [mms[q,0,0,1::2] for q in range(qs)])
	print(dataset)
	print('-' * len(dataset))
	print('       HKSmin WKS    HKSd5  HKSd10 MMRmin MMDmin MMRd5  MMDd5  MMRd10 MMDd10')
	print('mean: {}'.format(np.round(m, 4)))
	print('  sd: {}'.format(np.round(sd, 4)))
	print('\n')