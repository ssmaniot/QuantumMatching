# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

data = np.load("MAT/result_houses_full_reduced_thr.npz")

outcomesHW = data['outcomeHW']
outcomesMMS = data['outcomeMMS']
thresholds = data['thresholds']
quantiles = data['quantiles']

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = thresholds
Y = quantiles
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
Z = np.transpose(outcomesMMS[0, ..., 0])

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel(r"Threshold")
ax.set_ylabel(r"$Quantile$")

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_zlabel(r"Avg. Accuracy MMS (%)")

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
