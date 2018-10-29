from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraypad import _validate_lengths


clamped = (1, 0.0)
free = (2, 0.0)
extrap = (-1, 0)
periodic = (-1, -1.0)

bc_map: {clamped: clamped, free: free, extrap: None, periodic: None}

class NDBPoly(object):
    def __init__(self, y, x, bcs=None):
        if isinstance(x, np.ndarray): # mesh
            self.x_mesh = x
            if x.ndim == 1:
                self.ndim = 1
            else:
                self.ndim = x.ndim - 1
            self.x_vectors = tuple(np.unique(self.x_mesh[k,...])
                for k in range(self.ndim))

        elif not isinstance(x, str) and len(x): # vectors, or try
            self.x_vectors = x
            self.x_mesh = self.meshgrid(x, indexing='ij')
            self.ndim = len(x)
        else:
            raise ValueError("Don't know how to interpret x")
        
        if y.ndim == self.ndim:
            # how can you tell if y needs a [None, ...] or [...] ?
            # same question as to whether ndim is y.ndim or y.ndim-1
            # I think this is the right answer. 
            self.y = y[None, ...]
        elif y.ndim == self.ndim+1:
            self.y = y
        else:
            raise ValueError("incompatible dimension size")
        self.mdim = self.y.shape[0]
        # generally, x.shape = (n1, n2, ..., n_ndim)
        # and y.sahpe = (mdim, n1, n2, ..., n_ndim)
        
        self.bcs = np.broadcast_to(bcs, (self.ndim,2,2))
        x_shape = np.r_[self.x_mesh.shape]
        deriv_specs = (self.bcs[:,:,0]!=-1).sum(axis=1)
        knot_shape = x_shape + 4 + deriv_specs

        self.knots = np.zeros(knot_shape)
        self.coeffs = np.pad(y, np.r_[np.c_[0,0], deriv_specs], 'constant')
        def get_coeffs_line(x, y):
            spline = interpolate.make_interp_spline(x, y)
            return spline.c
        def get_coeffs_1d():
            return
        for i in range(self.ndim):
            all_other_ax_shape = np.r_[self.x_mesh.shape[:i],
                self.x_mesh.shape[i+1:]]
            for idx in np.ndindex(all_other_ax_shape):
                line_sel = (Ellipsis,) + idx[:i] + (slice(None,None),)+ idx[i:]
                x_fit = self.x_vectors[j]
                axis_sel = (None,)*(self.ndim+self.mdim+i) +  + (None,)*(self.ndim-1-i)
                line_spline = interpolate.make_interp_spline(self.x[line_sel], self.y[line_sel].T, bc_type=bc_map[self.bcs[i]])
                self.knots[line_sel] = line_spline.t
                self.coefs[line_sel] = line_spline.c

x = np.r_[-1:1:5j]*np.pi/2
y = np.r_[-1:1:7j]*np.pi/2
meshx, meshy = np.meshgrid(x,y, indexing='ij')
input_coords = np.r_['0,3', meshx, meshy]
fvals = np.sin(meshx)*np.sin(meshy)# np.sqrt(meshx**2 + meshy**2)

spline = NDBPoly(fvals, input_coords)
