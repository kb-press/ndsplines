from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np

# ndimage.spline_filter1d()

xmax = 6
factor=0.125
x = np.unique(np.r_[0:xmax:(xmax+1)*1j])
xx = np.r_[-factor*xmax:(1+factor)*xmax:1024j].reshape(1,-1)

tck = interpolate.splprep(x=np.ones_like(x).reshape(1,-1), u=x.squeeze(), s=0)[0]
t = tck[0]
k = tck[2]
c_max = t.size-4

plt.figure()
for idx in range(c_max):
    c = np.r_[0:0:idx*1j, 1, 0:0:(c_max-idx-1)*1j]
    spline_basis = interpolate.splev(xx, (t,c,k))
    plt.plot(xx[0,:], spline_basis[0,:], '-C'+str(idx))
    
    # out = ndimage.map_coordinates(c, xx, prefilter=False, order=2, mode='nearest')
    # plt.plot(xx.squeeze(), out.squeeze(), '--C'+str(idx))
plt.plot(x, np.zeros_like(x), 'kx')
plt.show()

