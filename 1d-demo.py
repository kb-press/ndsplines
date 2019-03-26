from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os
import itertools

os.chdir(os.path.expanduser('~/Projects/ndsplines'))

import NdBPoly
import ndimage_ndpoly


importlib.reload(NdBPoly)
importlib.reload(ndimage_ndpoly)

x = np.r_[-1:1:9j] * np.pi
x = np.r_[-1:-0.5:3j, 0, 0.5:1:3j] *np.pi
fvals = np.sin(x)

k = 3
factor = 1.25
xx = np.r_[-1:1:1024j]*factor*np.pi

# ndimage_coef = ndimage.spline_filter(fvals)
# ndimage_out = ndimage.map_coordinates(ndimage_coef, ((xx-x[0])/(x[1]-x[0]))[None,...], prefilter=False)

ndimg_poly = ndimage_ndpoly.NdBPoly(fvals, x)
ndimage_out = ndimg_poly.evaluate(xx)

test_bcs = list(itertools.chain(
    itertools.product(["natural", "clamped"], repeat=2),
    ((None,None),),
))
NDspline_dict = {"natural": NdBPoly.pinned, "clamped": NdBPoly.clamped, None: -1}
NDspline_dict = {"natural": NdBPoly.pinned, "clamped": NdBPoly.clamped, None: 0}
skip_size = 32
plt.figure()

# plt.plot(xx, ndimage_out[0,...])
for test_bc in test_bcs:
    test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=test_bc)
    
    splinef = test_Bspline(xx)
    plt.plot(xx[::skip_size], splinef[::skip_size],'x')
    
    
plt.gca().set_prop_cycle(None)

for test_bc in test_bcs:
    test_NDBspline = NdBPoly.make_interp_spline(x, fvals, bcs=(NDspline_dict[test_bc[0]], NDspline_dict[test_bc[1]]))
    NDsplienf = test_NDBspline(xx)
    plt.plot(xx, NDsplienf.squeeze(), )
    

plt.plot(x, fvals, 'o')

plt.show()


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