import numpy as np

real = ["car", "houses_full_reduced", "moto", "socnet"]
synthetic = ["delaunay_graphs", "knn_k3_graphs", "knn_k4_graphs", "knn_k5_graphs", "scale_free_graphs", "small_world_k4_graphs", "small_world_k5_graphs"]
# methods = ['HKSmin', 'WKS', 'HKSd5', 'HKSd10', 'MMSrow min', 'MMSdiag min', 'MMRd5', 'MMDd5', 'MMRd10', 'MMDd10']
methods = ['HKS', 'WKS', 'MMSrow min', 'MMSdiag min', 'MMSrow d5', 'MMSdiag d5', 'MMSrow d10', 'MMSdiag d10']

with open('table.tex', 'w') as f:
	f.write('\\hline\n')
	f.write('Dataset')
	for method in methods:
		f.write(f' & {method}')
	f.write(r'\\')
	f.write('\n')
	f.write('\\hline\\hline\n')
	
	for dataset in real:
		f.write(dataset.replace('_', '-'))
		dmin = np.load('MAT/result_{}_dmin.npy'.format(dataset))
		d5   = np.load('MAT/result_{}_d5.npy'.format(dataset))
		d10  = np.load('MAT/result_{}_d10.npy'.format(dataset))
		# data = np.hstack([dmin[:,:4], d5[:,:2], d10[:,:2], dmin[:,4:], d5[:,4:], d10[:,4:]])
		data = np.hstack([dmin[:,:4], dmin[:,4:], d5[:,4:], d10[:,4:]])
		m = 100 * np.mean(data[:,0::2], axis = 0)
		sd = 100 * np.mean(data[:,1::2], axis = 0)
		for i in range(len(methods)):
			f.write(' & {:.2f}$\pm${:.2f}'.format(m[i], sd[i]))
		f.write(r'\\')
		f.write('\n\\hline\n')
		
	for dataset in synthetic:
		f.write(dataset.replace('_', '-'))
		data = np.load('MAT/result_{}.npz'.format(dataset))
		hw = data['outcomeHW']
		mms = data['outcomeMMS']
		qs = data['quantiles'].shape[0]
		m =  100 * np.hstack([hw[0,0], hw[0,2]] + [mms[q,0,0,::2] for q in range(qs)])
		sd =  100 * np.hstack([hw[0,1], hw[0,3]] + [mms[q,0,0,1::2] for q in range(qs)])
		for i in range(len(methods)):
			f.write(' & {:.2f}$\pm${:.2f}'.format(m[i], sd[i]))
		f.write(r'\\')
		f.write('\n\\hline\n')