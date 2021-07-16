import numpy as np
import torch
import math
from sklearn.utils.extmath import cartesian
import matplotlib.pyplot as plt

x_coord = torch.from_numpy(cartesian([np.arange(10),
                                      np.arange(10)]))

y_coord = torch.from_numpy(cartesian([np.arange(10),
                                      np.arange(10)]))

y_coord = torch.from_numpy(np.array([[0, 1], [1, 2], [2, 2], [8, 1], [1, 7], [7, 9], [9, 4], [8, 8]]))


def cdist(x, y, r=2):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences**r, -1) ** (1/r)
    return distances


d_matrix = cdist(x_coord, y_coord)
d_matrix = d_matrix.numpy()
# 10000, 10000

# sampling
distance_points = []
for i in range(100):
    for j in range(100):
        d = d_matrix[j][i]
        distance_points.append([j, i, d])

# d_max
d_max = np.max(d_matrix)
print(d_max)
distance_points = np.array(distance_points)
fig = plt.figure()
ax = fig.gca(projection='3d')

# axis scale setting
ax.set_xlim3d(0, 100)
ax.set_ylim3d(0, 100)
ax.set_zlim3d(-20, 20)

# ax.scatter(distance_points[:, 0],
#            distance_points[:, 1],
#            d_max,
#            alpha=1,
#            depthshade=False,
#            edgecolors=None,
#            )

ax.scatter(distance_points[:, 0],
           distance_points[:, 1],
           d_max - distance_points[:, 2],
           alpha=1,
           depthshade=False,
           edgecolors=None,
           )

# x, y, z = label
#
# # label
# ax.grid(False)
# ax.plot([0, scale * x], [0, scale * y], [0, scale * z])
#
# # how rotate they are
# phi2 = np.arctan2(y, x) * 180 / np.pi
# theta = np.arccos(z) * 180 / np.pi
#
# if phi2 < 0:
#     phi2 = 360 + phi2

# ax.set_aspect('equal')
# rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
# rgb = np.concatenate([rgb, rgb, rgb], axis=1)
#
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=1, facecolors=rgb, depthshade=False,
#            edgecolors=None,
#            )  # data coloring

plt.legend(loc=2)
# Photos viewed at 90 degrees
# ax.view_init(-1 * theta, phi2)
#
# # Photos from above
# ax.view_init(-1 * theta + 90, phi2)

plt.draw()
plt.show()

print(d_matrix)