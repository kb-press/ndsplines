from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
import NdBPoly
import importlib

importlib.reload(NdBPoly)

# x,y = ogrid[-np.pi:np.pi:50j,-np.pi:np.pi:5j]
x = np.r_[-1:1:4j]*np.pi/2
y = np.r_[-1:1:5j]*np.pi/2
meshx, meshy = np.meshgrid(x,y, indexing='ij')
input_coords = np.r_['0,3', meshx, meshy]
fvals = np.sin(meshx)*np.sin(meshy)# np.sqrt(meshx**2 + meshy**2)
test_NDBspline = NdBPoly.NDBPoly(input_coords, fvals)

factor = 1
newx = np.r_[-1:1:10j]*factor*np.pi/2
newy = np.r_[1:1:1j]*factor*np.pi/2
newmeshx, newmeshy = np.meshgrid(newx,newy, indexing='ij')
newxymesh = np.r_['0,3', newmeshx, newmeshy]

splinef = test_NDBspline(newxymesh)

##



np.allclose(splinef, np.sin(newmeshx)*np.sin(newmeshy))

plt.figure()
# plt.imshow(splinef.prod(axis=2)[0,...])
plt.plot(newx, splinef.prod(axis=1)[0,:,0])
plt.show()



##

testpoly = NdBPoly.NdBPoly(fvals, input_coords, odd_extrapolation=True)
# testpoly.transform_coord_to_pixel(np.r_[-1.0, 0.05])

testspline = interpolate.make_interp_spline(input_coords[0,:,:], fvals, axis=1)

factor = 1
newx = np.r_[-1:1:500j]*factor*np.pi/2
newy = np.r_[-1:1:300j]*factor*np.pi/2

# newx = np.r_[-1:1:11j]*factor*np.pi/2
# newy = np.r_[-1:1:11j]*factor*np.pi/2
newmeshx, newmeshy = np.meshgrid(newx,newy, indexing='ij')
newxymesh = np.r_['0,3', newmeshx, newmeshy]
indices = testpoly.indices_from_coords(newxymesh)
# print(indices)
# new_coords, aliasing_mask, dxs = testpoly.transform_coord_to_pixel(newxymesh)
newf = testpoly.evaluate(newxymesh)



plt.figure()
plt.plot(x, fvals[:,-1], 'o')
plt.plot(newx, newf[0,:,-1])
plt.show()

plt.figure()
plt.imshow(newf[0])
plt.show()

plt.figure()
plt.imshow(fvals)
plt.show()