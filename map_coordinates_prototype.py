from scipy import ogrid, sin, mgrid, ndimage, array, interpolate
import matplotlib.pyplot as plt
import numpy as np
import ndimage_ndpoly as NdBPoly
import importlib

importlib.reload(NdBPoly)
##

        
                

# x,y = ogrid[-np.pi:np.pi:50j,-np.pi:np.pi:5j]
x = np.r_[-1:1:5j]*np.pi/2
y = np.r_[-1:1:7j]*np.pi/2
meshx, meshy = np.meshgrid(x,y, indexing='ij')
input_coords = np.r_['0,3', meshx, meshy]
fvals = sin(meshx)*sin(meshy)

testpoly = NdBPoly.NdBPoly(fvals, input_coords)
# testpoly.transform_coord_to_pixel(np.r_[-1.0, 0.05])

factor = 5
newx = np.r_[-1:1:500j]*factor*np.pi/2
newy = np.r_[-1:1:300j]*factor*np.pi/2

# newx = np.r_[-1:1:11j]*factor*np.pi/2
# newy = np.r_[-1:1:11j]*factor*np.pi/2
newmeshx, newmeshy = np.meshgrid(newx,newy, indexing='ij')
newxymesh = np.r_['0,3', newmeshx, newmeshy]
indices = testpoly.indices_from_coords(newxymesh)
print(indices)
new_coords, aliasing_mask = testpoly.transform_coord_to_pixel(newxymesh)
newf = ndimage.map_coordinates(testpoly.coeffs[0], new_coords, prefilter=False, mode='nearest')


plt.figure()
plt.imshow(newf)
plt.show()

plt.figure()
plt.imshow(fvals)
plt.show()
##


test_list = np.c_[
    np.r_[-1.0, 1.6],
    np.r_[-1.5, -0.5],
    np.r_[1.5, 0.5],
    np.r_[1.5, 0.5],
]

testx = np.r_[-0.9:0.9:5j]*np.pi
testy = np.r_[-0.9:0.9:7j]*np.pi

# 
# print(testpoly.transform_coord_to_pixel(test_list[:,0]))
# print(testpoly.transform_coord_to_pixel(test_list[:,0, None]))
# print(testpoly.transform_coord_to_pixel(test_list))
# print(testpoly.transform_coord_to_pixel(np.r_['0,3', np.meshgrid(testx,testy)]))


# ge_test = testpoly.knots > np.r_[-1.0, 0.05][(Ellipsis,)+2*(np.newaxis,)]
# for k in range(self.ndim).sf
    
##
        
newx = np.r_[-1:1:500j]*np.pi
newy = y
newmeshx, newmeshy = np.meshgrid(newx,newy)
x0 = meshx[0,0]
y0 = meshy[0,0]
dx = meshx[1,0] - x0
dy = meshy[0,1] - y0
ivals = (newmeshx - x0)/dx
jvals = (newmeshy - y0)/dy
coords = array([ivals, jvals])
# newf = ndimage.map_coordinates(fvals, coords)

coeffs = ndimage.spline_filter(fvals, order=3)

# spline_filter is just spline_filter1d on each axis, order doesn't matter
ndimage.spline_filter1d(ndimage.spline_filter1d(fvals, order=3, axis=0), order=3, axis=1) - coeffs
ndimage.spline_filter1d(ndimage.spline_filter1d(fvals, order=3, axis=0), order=3, axis=1) - coeffs

# interpolate.NdPPoly(coeffs, (meshx.reshape(-1), meshy.reshape(-1)))

newf = ndimage.map_coordinates(testpoly.coeffs, testpoly.transform_coord_to_pixel(np.r_['0,3', newmeshx, newmeshy]), prefilter=False)

##
plt.figure();
idx = 1
plt.plot(x,fvals[idx,:], 'o')
plt.plot(newx, newf[idx,:])
plt.show()

##
# x=y=np.r_[-1:1:5j]
# meshx, meshy = np.meshgrid(x,y)
# oldxy = np.r_['0,3',meshx,meshy]
# fvals = np.sin(meshx)*np.sin(meshy)

# 

# coeffs = ndimage.spline_filter(fvals, order=3)
# ndimage.spline_filter1d(ndimage.spline_filter1d(fvals, order=3, axis=0), order=3, axis=1) - coeffs
# ndimage.spline_filter1d(ndimage.spline_filter1d(fvals, order=3, axis=0), order=3, axis=1) - coeffs
# ndimage.map_coordinates(coeffs, oldxy, prefilter=False) - fvals


xderiv = (x[:-1] + x[1:])/2
yderiv = (y[:-1] + y[1:])/2

meshx_deriv, meshy_deriv = np.meshgrid(xderiv,yderiv)

coeffs_dx = ndimage.spline_filter1d(ndimage.spline_filter1d(fvals, order=2, axis=0), order=3, axis=1)
coeffs_dy = ndimage.spline_filter1d(ndimage.spline_filter1d(fvals, order=3, axis=1), order=2, axis=0)

newf_x = ndimage.map_coordinates(coeffs_dx, np.r_['0,3', np.meshgrid(x[1:],yderiv)], prefilter=False) - ndimage.map_coordinates(coeffs_dx, np.r_['0,3', np.meshgrid(x[:-1],yderiv)], prefilter=False)



dt = 2**-2
num_deriv = (ndimage.map_coordinates(coeffs, np.r_['0,3',np.meshgrid(xderiv+dt,yderiv)], prefilter=False)-ndimage.map_coordinates(coeffs, np.r_['0,3',np.meshgrid(xderiv-dt,yderiv)], prefilter=False))/(2*dt)




fderiv = -np.sin(meshx_deriv)*np.cos(meshy_deriv)
