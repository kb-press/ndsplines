from scipy import ogrid, sin, mgrid, ndimage, array, interpolate
import matplotlib.pyplot as plt
import numpy as np
import NdBPoly
import importlib

importlib.reload(NdBPoly)


        
                

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
new_coords, aliasing_mask, dxs = testpoly.transform_coord_to_pixel(newxymesh)
newf = testpoly.evaluate(newxymesh)


plt.figure()
plt.imshow(newf[0])
plt.show()

plt.figure()
plt.imshow(fvals)
plt.show()