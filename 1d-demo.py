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

test_Bspline = interpolate.make_interp_spline(x, fvals)


factor = 3
xx = np.r_[-1:1:500j]*factor*np.pi/2

splinef = test_Bspline(xx)
NDsplienf = test_NDBspline(xx[None,...])
##
textx = xx[None,...]
indices = test_NDBspline.indices_from_coords(textx)
print(np.all(test_NDBspline.knots[0,indices] <= textx))
##
plt.figure()
t = test_NDBspline.knots.squeeze()
k = 3
factor = 1.125
xx = np.r_[-1:1:500j]*factor*np.pi

tck = interpolate.splprep(x=np.ones_like(x).reshape(1,-1), u=x.squeeze(), s=0)[0]
c_max = t.size-4
for idx in range(c_max):
    c = np.r_[0:0:idx*1j, 1, 0:0:(c_max-idx-1)*1j]
    # c = np.ones_like(test_NDBspline.coeffs.squeeze())
    spline_basis = interpolate.splev(xx, (t,c,k))
    plt.plot(xx, spline_basis)
plt.show()
##
plt.figure()

x = np.unique(np.r_[-1:1:9j]) * np.pi
fvals = np.sin(x)

test_Bspline = interpolate.make_interp_spline(x, fvals)
test_NDBspline = NdBPoly.NDBPoly(x, fvals)

NDsplienf = test_NDBspline(xx)

tck = interpolate.splprep(x=fvals.reshape(1,-1), u=x, s=0)[0]

k = 2
factor = 2.0
xx = np.r_[-1:1:500j]*factor*np.pi
splinef = test_Bspline(xx)

# for idx in range(NDsplienf[0].shape[0]):
    # plt.plot(xx, NDsplienf[0][idx,:])
# plt.plot(xx[1::3], interpolate.splev(xx.reshape(25,-1), test_Bspline.tck).reshape(1,-1).squeeze()[1::3], 'x')
# plt.plot(xx[::3], splinef.squeeze()[::3], 'x')

plt.plot(xx, (test_NDBspline.coeffs@NDsplienf[0])[0,:], 'o')

plt.plot(x, fvals, 'o')

plt.show()


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
    print(test_Bspline.t)
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