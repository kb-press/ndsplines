"""
===============================
2-Dimensional Least Squares Fit
===============================
"""

import ndsplines 
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


NUM_X = 50
NUM_Y = 50
x = np.linspace(-3, 3, NUM_X)
y = np.linspace(-3, 3, NUM_Y)
meshx, meshy = np.meshgrid(x,y, indexing='ij')
input_coords = np.stack((meshx, meshy), axis=-1)
K = np.array([[1, -0.7,], [-0.7, 1.5]])
meshz = np.exp(-np.einsum(K, [1,2,], input_coords, [...,1], input_coords, [...,2])) + 0.1 * np.random.randn(NUM_X,NUM_Y)


xt = [-1, 0, 1]
yt = [-1, 0, 1]
k = 3
xt = np.r_[(x[0],)*(k+1),
          xt,
          (x[-1],)*(k+1)]
yt = np.r_[(y[0],)*(k+1),
          yt,
          (y[-1],)*(k+1)]
ts = [xt, yt]

spl = ndsplines.make_lsq_spline(input_coords.reshape((-1,2)), meshz.reshape((-1,1)), ts, np.array([3,3]))
# spl.coefficients = spl.coefficients.swapaxes(0,1)
# plt.figure()
# plt.contour(meshx, meshy, meshz)
# plt.contour(meshx, meshy, spl(input_coords)[...,0])
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(meshx, meshy, meshz, alpha=0.25)
ax.plot_wireframe(meshx, meshy, spl(input_coords), color='C1')
plt.show()
