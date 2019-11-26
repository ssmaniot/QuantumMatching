# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LightSource
import numpy as np

synth = ["small_world_k{}_graphs".format(k) for k in range(2, 6)] # ["delaunay_graphs", "knn_k3_graphs", "knn_k4_graphs", "knn_k5_graphs", "scale_free_graphs", "small_world_graphs"]
for dataset in synth:
	data = np.load("MAT/result_{}.npz".format(dataset))

	outcomesHW = data['outcomeHW']
	outcomesMMS = data['outcomeMMS']
	t_max = data['t_max']
	thresholds = data['thresholds']
	quantiles = data['quantiles']

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Make data.
	X = t_max
	Y = quantiles
	X, Y = np.meshgrid(X, Y)
	# R = np.sqrt(X**2 + Y**2)
	Z = np.transpose(np.swapaxes(outcomesMMS, 0, 1)[..., 0, 2])

	# Plot the surface.
	ls = LightSource(270, 45)
	m = cm.gist_earth#viridis
	rgb = ls.shade(Z, cmap=m, vert_exag=0.1, blend_mode='soft')
	surf = ax.plot_surface(X, Y, Z, facecolors=rgb, # cmap=cm.coolwarm,
	                       linewidth=0, antialiased=False)
	ax.set_xlabel(r"$t_{\max}$")
	ax.set_ylabel(r"First $q$-quantiles")

	# Customize the z axis.
	# ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	ax.set_zlabel(r"Avg. Accuracy MMS (%)")

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.title("MMS row on {}".format(dataset))
	plt.yticks(range(quantiles[0], quantiles[-1] + 1))
	plt.savefig("Figures/{}.pdf".format(dataset))
	plt.show()
