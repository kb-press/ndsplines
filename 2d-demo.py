from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
import importlib
import os

os.chdir(os.path.expanduser('~/Projects/ndsplines'))

import NDBSpline
import ndimage_ndpoly


importlib.reload(NDBSpline)
importlib.reload(ndimage_ndpoly)

# x,y = ogrid[-np.pi:np.pi:50j,-np.pi:np.pi:5j]
x = np.r_[-1:1:5j]*np.pi/2
y = np.r_[-1:1:5j]*np.pi/2
meshx, meshy = np.meshgrid(x,y, indexing='ij')
input_coords = np.r_['0,3', meshx, meshy]
fvals = np.sin(meshx)*np.sin(meshy)# np.sqrt(meshx**2 + meshy**2)


factor = 1.25
newx = np.r_[-1:1:1024j]*factor*np.pi/2
newy = np.r_[-1:1:5j]*np.pi/2
newmeshx, newmeshy = np.meshgrid(newx,newy, indexing='ij')
newxymesh = np.r_['0,3', newmeshx, newmeshy]

truef = np.sin(newmeshx)*np.sin(newmeshy)

# np.allclose(splinef, truef)
skip_size = 32

test_NDBspline = NDBSpline.make_interp_spline(input_coords, fvals, bcs=(NDBSpline.clamped))
# new API for extrapolate/BC behavior requires overriding to match scipy.interpolate.make_interp_spline behavior
test_NDBspline.extrapolate = np.ones_like(test_NDBspline.extrapolate)

print(np.allclose(test_NDBspline(input_coords),fvals))

plt.figure()
plt.plot(x, np.zeros_like(x), 'kx')
plt.plot(newx[::skip_size], truef[::skip_size,:], 'x')
plt.gca().set_prop_cycle(None)

for yidx, yy in enumerate(y):
    test_Bspline = interpolate.make_interp_spline(x, fvals[:,yidx], k=3, bc_type="clamped")
    bsplinef = test_Bspline(newx, extrapolate=False)
    plt.plot(newx, bsplinef, 'k--', lw=2.0)

plt.gca().set_prop_cycle(None)
splinef = test_NDBspline(newxymesh)
plt.plot(newx, splinef[0,:,:], alpha=0.75)

plt.show()
