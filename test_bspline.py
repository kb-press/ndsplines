from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np

import NdBPoly
import importlib
import itertools
importlib.reload(NdBPoly)


## testing what different BC's do
test_bcs = itertools.chain(
    
    itertools.product(["natural", "clamped"], repeat=2),
    ((None,None),),
)
x = np.unique(np.r_[-1:1:9j]) * np.pi
fvals = np.sin(x)
kval = 3
print("k=",kval,"x.shape:",x.shape)
coeffs = ndimage.spline_filter1d(fvals)
tck = interpolate.splprep(x=fvals.reshape(1,-1), u=x, s=0)
print("splprep, t:", tck[0][0].shape, "c", tck[0][1][0].shape)
for test_bc in test_bcs:
    test_Bspline = interpolate.make_interp_spline(x, fvals, k=kval, bc_type=test_bc)
    print(test_bc, "t.shape:", test_Bspline.t.shape, "c.shape:", test_Bspline.c.shape)
    print("knots:")
    print(test_Bspline.t)
    print("coeff difference:") # the internal knots of clamped-clamped are the same as ndimage.spline_filter
    if test_Bspline.c.size == coeffs.size:
        print(test_Bspline.c - coeffs)
    else:
        print(test_Bspline.c[1:-1] - coeffs)
    
##
xmax = 9
x = np.r_[0:xmax:(xmax+1)*1j]
fvals = np.sin((2*x/xmax-0.5)*np.pi)

x = np.r_[-1:1:xmax*(1j)]*np.pi
fvals = np.sin(x)

kval = 3
test_bc = None

test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=test_bc)
knot_gen = interpolate.make_interp_spline(x, np.zeros_like(x), bc_type=test_bc)

res = interpolate.splprep(x=fvals.reshape(1,-1), u=x, s=0)
tck = res[0]

# plt.plot(xx[1::3], interpolate.splev(xx.reshape(25,-1), test_Bspline.tck).reshape(1,-1).squeeze()[1::3], 'x')
# plt.plot(xx[::3], splinef.squeeze()[::3], 'x')


print(test_Bspline.t - knot_gen.t) # knots are independent of coeff
print(test_Bspline.c - tck[1])
print(test_Bspline.t - tck[0])

## testing indices_from_coords

textx = xx[None,...]
indices = test_NDBspline.indices_from_coords(textx)
print(np.all(test_NDBspline.knots[0,indices] <= textx))