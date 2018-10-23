from scipy import ogrid, sin, mgrid, ndimage, array, interpolate
import matplotlib.pyplot as plt
import numpy as np
import NdBPoly
import importlib

importlib.reload(NdBPoly)

# x,y = ogrid[-np.pi:np.pi:50j,-np.pi:np.pi:5j]
x = np.unique(np.r_[-1:-1/3:4j, -1/3:1/3:3j, 1/3:1:4j]) * np.pi
fvals = np.sin(x)

testpoly = NdBPoly.NdBPoly(fvals, x.reshape(1,-1), odd_extrapolation=True)
# testpoly.transform_coord_to_pixel(np.r_[-1.0, 0.05])

factor = 2
xx = np.r_[-1:1:500j]*factor*np.pi/2

# indices = testpoly.indices_from_coords(xx)
# print(indices)
# new_coords, aliasing_mask, dxs = testpoly.transform_coord_to_pixel(xx)
newf = testpoly.evaluate(xx.reshape(1,-1))

plt.figure()
plt.plot(xx, newf.squeeze(), 'o')
plt.plot(x, fvals, 'o')
plt.show()