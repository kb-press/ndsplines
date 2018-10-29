from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
import NdBPoly
import importlib
import itertools
importlib.reload(NdBPoly)


# x = np.unique(np.r_[-1:-1/3:4j, -1/3:1/3:3j, 1/3:1:4j]) * np.pi
x = np.unique(np.r_[-1:1:9j]) * np.pi
fvals = np.sin(x)

testpoly = NdBPoly.NdBPoly(fvals, x.reshape(1,-1), odd_extrapolation=True)
testpoly.transform_coord_to_pixel(np.r_[-1.0, 0.05])


##
test_bcs = itertools.chain(
    
    itertools.product(["natural", "clamped"], repeat=2),
    ((None,None),),
)
x = np.unique(np.r_[-1:1:9j]) * np.pi
fvals = np.sin(x)
kval = 3
print("k=",kval,"x.shape:",x.shape)
tck = interpolate.splprep(x=fvals.reshape(1,-1), u=x, s=0)
print("splprep, t:", tck[0][0].shape, "c", tck[0][1][0].shape)
for test_bc in test_bcs:
    test_Bspline = interpolate.make_interp_spline(x, fvals, k=kval, bc_type=test_bc)
    print(test_bc, "t.shape:", test_Bspline.t.shape, "c.shape:", test_Bspline.c.shape)
    ##
    

test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=test_bc)


test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=test_bc)
knot_gen = interpolate.make_interp_spline(x, np.zeros_like(x), bc_type=test_bc)

print(test_Bspline.t - knot_gen.t)


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