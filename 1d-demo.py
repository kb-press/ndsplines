from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
import NdBPoly
import importlib
import itertools

importlib.reload(NdBPoly)
plt.figure()

x = np.unique(np.r_[-1:1:9j]) * np.pi
fvals = np.sin(x)

test_Bspline = interpolate.make_interp_spline(x, fvals)
test_NDBspline = NdBPoly.NDBPoly(x, fvals)


tck = interpolate.splprep(x=fvals.reshape(1,-1), u=x, s=0)[0]

k = 3
factor = 2.0
xx = np.r_[-1:1:500j]*factor*np.pi
splinef = test_Bspline(xx)
NDsplienf = test_NDBspline(xx)

ndimage_coef = ndimage.spline_filter(fvals)
ndimage_out = ndimage.map_coordinates(ndimage_coef, ((xx-x[0])/(x[1]-x[0]))[None,...], prefilter=False)

# plt.plot(xx[1::3], interpolate.splev(xx.reshape(25,-1), test_Bspline.tck).reshape(1,-1).squeeze()[1::3], 'x')
# plt.plot(xx[::3], splinef.squeeze()[::3], 'x')

plt.plot(xx, NDsplienf.squeeze(), 'o')
plt.plot(xx, ndimage_out)
plt.plot(x, fvals, 'o')

plt.show()


##

    ##
    




factor = 2
xx = np.r_[-1:1:500j]*factor*np.pi/2
# xx = np.unique(np.r_[-1:1:10j]) * np.pi
indices = testpoly.indices_from_coords(xx)
# print(indices)
# new_coords, aliasing_mask, dxs = testpoly.transform_coord_to_pixel(xx)
newf = testpoly.evaluate(xx.reshape(1,-1))
splinef = test_Bspline(xx)

testpoly.coeffs[0][1:-1] - test_Bspline.c[2:-2] # great, B-splines use the cardinal basis, so we good

plt.figure()
plt.plot(xx, newf.squeeze(), 'o')
plt.plot(xx, splinef.squeeze(), 'x')
plt.plot(x, fvals, 'x')
plt.show()