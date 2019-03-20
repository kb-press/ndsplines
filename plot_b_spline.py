from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
import os, sys

os.chdir(os.path.expanduser('~/Projects/ndsplines'))

# ndimage.spline_filter1d()

xmax = 6
factor=0.125
x = np.unique(np.r_[0:xmax:(xmax+1)*1j])
xx = np.r_[-factor*xmax:(1+factor)*xmax:1024j].reshape(1,-1)

tck = interpolate.splprep(x=np.ones_like(x).reshape(1,-1), u=x.squeeze(), s=0)[0]
t = tck[0]
k = tck[2]
c_max = t.size-4
"""
plt.figure()
for idx in range(c_max):
    c = np.r_[0:0:idx*1j, 1, 0:0:(c_max-idx-1)*1j]
    spline_basis = interpolate.splev(xx, (t,c,k))
    plt.plot(xx[0,:], spline_basis[0,:], '-C'+str(idx))
    
    # out = ndimage.map_coordinates(c, xx, prefilter=False, order=2, mode='nearest')
    # plt.plot(xx.squeeze(), out.squeeze(), '--C'+str(idx))
plt.plot(x, np.zeros_like(x), 'kx')
plt.plot(t, np.arange(t.shape[0])*1/t.max(), 'C0o', alpha=0.5)
"""
import NdBPoly
import importlib

importlib.reload(NdBPoly)
ells = NdBPoly.find_intervals(t, xx.squeeze(), 3, True)
plt.step(xx.squeeze(), ells*1/t.max())
bases = NdBPoly.eval_bases(t, xx.squeeze(), ells, 3, 0)
bases2 = NdBPoly.process_bases_call(t, xx.squeeze(), 3, extrapolate=True)
"""
plt.figure()
for ell in np.unique(ells):
    for idx in range(4):
        plt.plot(xx[0,ells==ell], bases[idx, ells==ell])
plt.step(xx.squeeze(), ells*1/t.max())
plt.show()
"""
plt.figure()
for idx in range(4):
    
    plt.plot(xx[0], bases2[idx, :])
    plt.plot(xx[0], bases[idx, :] , '--')
plt.show()

##



plt.figure()

plt.show()

##
plt.figure()
c = np.eye(c_max)
bases = np.array(interpolate.splev(xx, (t,c,k)))
for idx in range(c_max):
    plt.plot(xx[0,:], bases[idx,0,:], '-C'+str(idx))
plt.plot(x, np.zeros_like(x), 'kx')
plt.show()

##
plt.figure()
c = np.zeros((2,3,7))
c[0,:,:3] = np.eye(3)
c[1,:,-3:] = np.eye(3)

bases = np.array(interpolate.splev(xx, (t,np.moveaxis(c,2,1),k)))

import scipy.interpolate._bspl as _bspl

out = np.empty((xx.size, 
# bases2 = _bspl.evaluate_spline(t,c.reshape((-1,7)),xx.squeeze(),k, True, out)

print(bases.shape)
for idx in np.ndindex(bases.shape[:-1]):
    plt.plot(xx[0,:], bases[idx + (slice(None),)],label=str(idx))# '-C'+str(idx))
plt.plot(x, np.zeros_like(x), 'kx')
plt.legend()
plt.show()
    