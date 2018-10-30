from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraypad import _validate_lengths
from functools import reduce


pinned = 2
clamped = 1
extrap = 0
periodic = -1
bc_map =  {clamped: "clamped", pinned: "natural", extrap: None, periodic: None}

class NDBPoly(object):
    def __init__(self, x, y, bcs=0):
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
        self.bcs = np.broadcast_to(bcs, (self.ndim,2))
        self.order = np.broadcast_to(3, (self.ndim,))
        knot_shape = np.r_[self.x.shape]
        deriv_specs = np.asarray((self.bcs[:,:]>0),dtype=np.int)
        knot_shape[1:] = knot_shape[1:] + 4 + deriv_specs.sum(axis=1)
        self.splines = np.empty(self.x.shape[1:], dtype=object)

        self.knots = np.zeros(knot_shape)
        if self.mdim == 1:
            self.coeffs = np.pad(y, deriv_specs, 'constant')[None,...]
        else:
            self.coeffs = np.pad(y, np.r_[np.c_[0,0], deriv_specs], 'constant')

        for i in np.arange(self.ndim)+1:
            all_other_ax_shape = np.asarray(np.r_[self.coeffs.shape[1:i],
                self.y.shape[i+1:]], dtype=np.int)
            x_line_sel = ((i-1,) + (0,)*(i-1) + (slice(None,None),) +
                (0,)*(self.ndim-i))
            xp = self.x[x_line_sel]
            order = self.order[i-1]
            for idx in np.ndindex(*all_other_ax_shape):
                offset_axes_remaining_sel = (tuple(idx[i-1:] + 
                    deriv_specs[i:,0]))
                y_line_sel = ((Ellipsis,) + idx[:i-1] + 
                    (slice(deriv_specs[i-1,0],-deriv_specs[i-1,0] or None),) +
                    offset_axes_remaining_sel)
                coeff_line_sel = ((Ellipsis,) + idx[:i-1] + (slice(None,None),)
                    + offset_axes_remaining_sel)
                line_spline = interpolate.make_interp_spline(xp,
                    self.coeffs[y_line_sel].T,
                    k = order,
                    bc_type=(bc_map[(self.bcs[i-1,0])],
                             bc_map[(self.bcs[i-1,1])]),

                )
                self.coeffs[coeff_line_sel] = line_spline.c.T
            self.knots[i-1,...] = (line_spline.t[(None,)*(i-1) + 
                (slice(None),) + (None,)*(self.ndim-i)])

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


        x = self.broadcast_coords(x)

        # TODO: IMPLEMENT PERIODIC BC
        # periodic_sel = np.all(self.bcs==-1, axis=1)
        # ns = np.r_[self.knots.shape[1:]] - self.order - 1
        # x[periodic_sel,...] = self.knots[periodic_sel, self.order[periodic_sel]]
        """
        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].        
        n = self.knots.shape[1:][periodic_sel] - self.order - 1
        x[] = self.t[self.k] + (x - self.t[self.k]) % (self.t[n] -
                                                     self.t[self.k])
        """
        u_mats = []
        y = np.zeros((self.mdim,) + x.shape[1:])
        input_op = list(range(self.ndim+1)) + [...,]
        u_ops = []
        output_op = [0,...]
        for i in np.arange(self.ndim)+1:
            nu = nus[i-1]
            order = self.order[i-1]
            knot_sel = ((i-1,) + (0,)*(i-1) + (slice(None,None),) + 
                (0,)*(self.ndim-i))
            ts = self.knots[knot_sel]
            xp = x[i-1,...]
            num_bases = self.coeffs.shape[i]
            u_mats.append(np.empty((num_bases,) + xp.shape))
            u_ops.append([int(i), ...])
            # cs = np.eye(num_bases)
            # us.append(interpolate.splev(xp, (ts,cs,self.order)))
            for j in range(num_bases):
                cs = np.r_[0:0:j*1j, 1.0, 0:0:(num_bases-j-1)*1j]
                tck = (ts,cs,order)
                u_mats[-1][j, ...] = interpolate.splev(xp, tck, nu)

        u_args = [subarg for arg in zip(u_mats, u_ops) for subarg in arg]
        y_out = np.einsum(self.coeffs, input_op, *u_args, output_op)
        return y_out
