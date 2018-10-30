from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraypad import _validate_lengths
from functools import reduce


# clamped = (1, 0.0)
# pinned = (2, 0.0)
# clamped_clamped = (clamped, clamped)
# pinned_pinned = (pinned, pinned)
# clamped_pinned = (clamped, pinned)
# pinned_clamped = (pinned, clamped)
# 
# extrap = (-1, -1)
# periodic = (-1, 0.0)
pinned = 2
clamped = 1
extrap = 0
periodic = -1
bc_map =  {clamped: "clamped", pinned: "natural", extrap: None, periodic: None}

# bc_map =  {clamped: clamped, pinned: pinned, extrap: None, periodic: None}
# bc_map[clamped_clamped] = clamped_clamped
# bc_map[pinned_pinned] = pinned_pinned
# bc_map[clamped_pinned] = clamped_pinned
# bc_map[pinned_clamped] = pinned_clamped

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
        self.order = 3
        self.bcs = np.broadcast_to(bcs, (self.ndim,2))
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
            x_line_sel = (i-1,) + (0,)*(i-1) + (slice(None,None),) + (0,)*(self.ndim-i)
            xp = self.x[x_line_sel]
            for idx in np.ndindex(*all_other_ax_shape):
                # line_sel_base = idx[:i-1] + (slice(None,None),)+ idx[i-1:]
                offset_axes_remaining_sel = tuple(idx[i-1:] + deriv_specs[i:,0])
                y_line_sel = (Ellipsis,) + idx[:i-1] + (slice(deriv_specs[i-1,0],-deriv_specs[i-1,0] or None),) + offset_axes_remaining_sel
                coeff_line_sel = (Ellipsis,) + idx[:i-1] + (slice(None,None),) + offset_axes_remaining_sel
                line_spline = interpolate.make_interp_spline(xp,
                    self.coeffs[y_line_sel].T,
                    bc_type=(bc_map[(self.bcs[i-1,0])],
                             bc_map[(self.bcs[i-1,1])])
                )
                self.coeffs[coeff_line_sel] = line_spline.c.T
            self.knots[i-1,...] = line_spline.t[(None,)*(i-1) + (slice(None),) + (None,)*(self.ndim-i)]

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
        # With periodic extrapolation we map x to the segment
        # [self.t[k], self.t[n]].

        # TODO: IMPLEMENT PERIODIC BC
        """
        periodic_sel = np.any(np.all(self.bcs==-1, axis=2), axis=1)
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
            data_all_other_ax_shape = np.asarray(np.r_[self.x.shape[1:i],
                self.x.shape[i+1:]], dtype=np.int)
            nu = nus[i-1]
            knot_sel = (i-1,) + (0,)*(i-1) + (slice(None,None),) + (0,)*(self.ndim-i)
            ts = self.knots[knot_sel]
            xp = x[i-1,...]
            num_bases = self.coeffs.shape[i]
            u_mats.append(np.empty((num_bases,) + xp.shape))
            u_ops.append([int(i), ...])
            # cs = np.eye(num_bases)
            # us.append(interpolate.splev(xp, (ts,cs,self.order)))
            for j in range(num_bases):
                cs = np.r_[0:0:j*1j, 1.0, 0:0:(num_bases-j-1)*1j]
                u_mats[-1][j, ...] = interpolate.splev(xp, (ts,cs,self.order), nu)

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
        u_args = [subarg for arg in zip(u_mats, u_ops) for subarg in arg]
        y_out = np.einsum(self.coeffs, input_op, *u_args, output_op)
        return y_out
