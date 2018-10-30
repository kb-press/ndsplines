from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraypad import _validate_lengths
import functools


clamped = (1, 0.0)
free = (2, 0.0)
extrap = (-1, 0)
periodic = (-1, -1.0)

bc_map =  {clamped: clamped, free: free, extrap: None, periodic: None}

class NDBPoly(object):
    def __init__(self, x, y, bcs=extrap):
        if isinstance(x, np.ndarray): # mesh
            self.x = x
            if x.ndim == 1:
                self.ndim = 1
                self.x = x[None,...]
            else:
                self.ndim = x.ndim - 1
            self.x_vectors = tuple(np.unique(self.x[k,...])
                for k in range(self.ndim))

        elif not isinstance(x, str) and len(x): # vectors, or try
            self.x_vectors = x
            self.x = self.meshgrid(x, indexing='ij')
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
        # generally, x.shape = (ndim, n1, n2, ..., n_ndim)
        # and y.sahpe = (mdim, n1, n2, ..., n_ndim)
        self.order = 3
        self.bcs = np.broadcast_to(bcs, (self.ndim,2,2))
        knot_shape = np.r_[self.x.shape]
        deriv_specs = np.asarray((self.bcs[:,:,0]!=-1),dtype=np.int)
        knot_shape[1:] = knot_shape[1:] + 4 + deriv_specs.sum(axis=1)
        self.splines = np.empty(self.x.shape[1:], dtype=object)

        self.knots = np.zeros(knot_shape)
        if self.mdim == 1:
            self.coeffs = np.pad(y, deriv_specs, 'constant')[None,...]
        else:
            self.coeffs = np.pad(y, np.r_[np.c_[0,0], deriv_specs], 'constant')

        for i in np.arange(self.ndim)+1:
            all_other_ax_shape = np.asarray(np.r_[self.x.shape[1:i],
                self.x.shape[i+1:]], dtype=np.int)
            for idx in np.ndindex(*all_other_ax_shape):
                line_sel_base = idx[:i-1] + (slice(None,None),)+ idx[i-1:]
                x_line_sel = (i-1,) + line_sel_base
                y_line_sel = (Ellipsis,) + line_sel_base
                
                line_spline = interpolate.make_interp_spline(self.x[x_line_sel],
                    self.y[y_line_sel].T,
                    bc_type=(bc_map[tuple(self.bcs[i-1,0,:])],
                             bc_map[tuple(self.bcs[i-1,1,:])])
                )
                self.knots[x_line_sel] = line_spline.t
                self.coeffs[y_line_sel] = line_spline.c.T
                self.splines[line_sel_base]

    def indices_from_coords(self, coords):
        indices = np.ones_like(coords, dtype=np.int_)*-1000
        vec_sel = (slice(None),) + (coords.ndim-1)*(None,)
        for k in range(self.ndim):
            knot_sel = (k,) + (0,)*(k) + (slice(None,None),) + (0,)*(self.ndim-k-1)
            lt_test = self.knots[knot_sel][vec_sel] >= coords[k, ...]
            ge_test =  coords[k, ...] >= self.knots[knot_sel][vec_sel]
            total_test = np.r_['0,{}'.format(coords.ndim-1), 
                (self.knots[knot_sel][vec_sel] >= 
                    coords[k, ...])[0,...][None,...],
                    
                lt_test[1:,...] & ge_test[:-1,...],
                
                (self.knots[knot_sel][vec_sel] <=
                    coords[k, ...])[-1,...][None,...],
            ]
            
            where_out = np.nonzero(total_test)
            indices[(k,)+where_out[1:]] = where_out[0] - 1
        return indices

    def broadcast_coords(self, coords):
        if coords.shape[0] == self.ndim:
            if coords.size == self.ndim:
                coords = coords.reshape((self.ndim, 1))
        elif coords.size%self.ndim == 0:
            coords = coords.reshape((self.ndim,-1))
        else:
            raise ValueError("Could not broadcast coords")
        return coords

    def __call__(self, x, nus=0):
        """
        Evaluate a spline function.

        Parameters
        ----------
        x : array_like
            points to evaluate the spline at.
        nu: int, optional
            derivative to evaluate (default is 0).

        Returns
        -------
        y : array_like
            Shape is determined by replacing the interpolation axis
            in the coefficient array with the shape of `x`.

        """
        nus = np.broadcast_to(nus, (self.ndim,))

        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].

        # TODO: IMPLEMENT PERIODIC BC
        """
        periodic_sel = np.any(np.all(self.bcs==-1, axis=2), axis=1)
        n = self.knots.shape[1:][periodic_sel] - self.order - 1
        x[] = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
                                                     self.t[self.k])
        """
        x = self.broadcast_coords(x)
        us = []
        y = np.zeros((self.mdim,) + x.shape[1:])

        for i in np.arange(self.ndim)+1:
            data_all_other_ax_shape = np.asarray(np.r_[self.x.shape[1:i],
                self.x.shape[i+1:]], dtype=np.int)
            nu = nus[i-1]
            knot_sel = (i-1,) + (0,)*(i-1) + (slice(None,None),) + (0,)*(self.ndim-i-1)
            ts = self.knots[knot_sel]
            xp = x[i-1,...]
            num_bases = self.coeffs.shape[i]
            us.append(np.empty((num_bases,) + xp.shape))
            for j in range(num_bases):
                cs = np.r_[0:0:j*1j, 1.0, 0:0:(num_bases-j-1)*1j]
                us[-1][j, ...] = interpolate.splev(xp, (ts,cs,self.order))

            """
            for idx in np.ndindex(*data_all_other_ax_shape):
                data_line_sel_base = idx[:i-1] + (slice(None,None),)+ idx[i-1:]
                t_line_sel = (i-1,) + data_line_sel_base
                c_line_sel = (Ellipsis,) + data_line_sel_base
                

                input_all_other_ax_shape = np.asarray(np.r_[x.shape[1:i],
                    x.shape[i+1:]], dtype=np.int)
                for inp_idx in np.ndindex(*input_all_other_ax_shape):
                    inp_line_sel_base = inp_idx[:i-1] + (slice(None,None),)+ inp_idx[i-1:]
                    x_line_sel = (i-1,) + inp_line_sel_base
                    y_line_sel = (Ellipsis,) + data_line_sel_base + inp_line_sel_base
                    xp = x[x_line_sel].copy()
                    out = np.empty((len(xp), self.mdim), dtype=self.coeffs.dtype)
                    # interpolate._bspl.evaluate_spline(ts.copy(), cs.copy(),
                    #     self.order, xp, nu, True, out)
                    y[y_line_sel] = out.reshape(xp.shape[0], -1).T
            """
        return us
        

x = np.r_[-1:1:5j]*np.pi/2
y = np.r_[-1:1:7j]*np.pi/2
meshx, meshy = np.meshgrid(x,y, indexing='ij')
input_coords = np.r_['0,3', meshx, meshy]
fvals = np.sin(meshx)*np.sin(meshy)# np.sqrt(meshx**2 + meshy**2)

# spline = NDBPoly(fvals, input_coords)
