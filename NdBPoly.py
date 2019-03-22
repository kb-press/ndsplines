from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraypad import _validate_lengths
from functools import reduce

"""
TODOs:

implement periodic call

implement a least-squares constructor

"""
pinned = 2
clamped = 1
extrap = 0
periodic = -1
bc_map =  {clamped: "clamped", pinned: "natural", extrap: None, periodic: None}

def find_intervals(t, k, x, extrapolate=False):
    """
    Find an interval ell such that t[ell] <= x < t[ell+1].

    Parameters
    ----------
    t : ndarray, shape=(n+k+1,) dtype=np.float_
        knots
    x : ndarray, shape=(s,) dtype=np.float_
        values to find the interval for
    k : int
        order of B-spline
    extrapolate : bool
        whether to return the last or the first interval if xval
        is out of bounds.

    Returns
    -------
    ell : ndarray, shape=(s,) dtype=np.int_
        Suitable interval or -1 for each value in x

    Similar to scipy\\interpolate\\_bspl.pyx find_Interval (inline)
    """
    test = (t[:-1,None] <= x[None,:]) & (x[None,:] < t[1:,None])
    ell = np.ones_like(x, dtype=np.int_)*-1
    if extrapolate:
        test[k,:] = test[k,:] | (x <= t[k])
        test[-k-1,:] = test[-k-1,:] | (t[-k-1] < x)
    where_test = np.where(test)
    ell[where_test[1]] = where_test[0]
    return ell

def eval_bases(t, k, x, ell, nu):
    """
    Evaluate the k+1 non-zero spline basis functions for x

    Parameters
    ----------
    t : ndarray, shape=(n+k+1,) dtype=np.float_
        knots
    x : ndarray, shape=(s,) dtype=np.float_
        values to find the interval for
    ell : ndarray, shape=(s,) dtype=np.int_
        index such that t[ell] <= x < t[ell+1] for each x
    k : int
        order of B-spline.
    nu : int
        order of derivative to evaluate.

    Returns
    -------
    u : ndarray, shape=(k+1, s) dtype=np.float_
        the values of the non-zero spline basis functions evaluated at x

    similar to scipy\\interpolate\\src\\__fitpack.h _deBoor_D (inline)
    """
    u = np.empty((k+1, x.size), dtype=np.float_)
    w = np.empty((k, x.size), dtype=np.float_)
    bounds = np.empty((2,x.size),)
    
    w[0,...] = 1.0
    for j in range(1, k-nu+1):

        u[0,:] = 0
        for n in range(1, j+1):
            index = ell+n
            bounds[0, :] = t[index]
            bounds[1, :] = t[index-j]
            neq_test = bounds[0, :] != bounds[1, :]
            u[n, ~neq_test] = 0.0

            # I'm not sure if using these views are cythonizable, but might
            # be faster in Python+Numpy?
            xb = bounds[0, neq_test]
            xa = bounds[1, neq_test]
            xx = x[neq_test]
            tau = w[n-1, neq_test]/(xb-xa)
            u[n-1, neq_test] += tau*(xb - xx)
            u[n, neq_test] = tau*(xx - xa)
            
        # w[:j] = u[:j]
        w[:] = u[:k].copy()

    for j in range(k-nu+1, k+1):
        u[0,:] = 0
        for n in range(1, j+1):
            index = ell+n
            bounds[0, :] = t[index]
            bounds[1, :] = t[index-j]
            neq_test = bounds[0, :] != bounds[1, :]
            u[n, ~neq_test] = 0.0 # __fitpack.h has h[m] = 0.0 here...

            xb = bounds[0, neq_test]
            xa = bounds[1, neq_test]
            xx = x[neq_test]
            tau = j*w[n-1, neq_test]/(xb-xa)
            u[n-1, neq_test] -= tau
            u[n, neq_test] = tau

        w[:] = u[:k].copy()

    return u


def process_bases_call(t, k, x, nu=0, periodic=False, extrapolate=True):
    """

    Similar to scipy\\interpolate\\_bsplines.py BSpline.__call__

    which goes through BSpline._evaluate and onto
    scipy\\interpolate\\_bspl.pyx evaluate_spline
    """
    x_shape, x_ndim = x.shape, x.ndim
    x = np.ascontiguousarray(x.ravel(), dtype=np.float_)

    if periodic:
        n = t.size - k - 1
        x = t[k] + (x - t[k]) % (t[n] - t[k])
        ell = find_intervals(t, k, x, False)
    else:
        ell = find_intervals(t, k, x, extrapolate)

    u = eval_bases(t, k, x, ell, nu)
    u.reshape((k+1,)+x_shape)

    return u


    

class NDBPoly(object):
    def __init__(self, knots, coeffs, orders=3, periodic=False, extrapolate=True):
        self.knots = knots
        self.coeffs = coeffs
        self.ndim = knots.shape[0] # dimension of knots
        self.mdim = coeffs.shape[0] # dimension of coefficeints
        self.orders = np.broadcast_to(orders, (self.ndim,))
        self.periodic = np.broadcast_to(periodic, (self.ndim,))
        self.extrapolate = np.broadcast_to(extrapolate, (self.ndim,))

        self.u_ops = []
        self.input_op = list(range(self.ndim+1)) + [...,]
        self.output_op = [0,...]
        self.tcks = []

        for i in np.arange(self.ndim)+1:
            self.u_ops.append([int(i), ...])
            num_bases = self.coeffs.shape[i]
            cs = np.eye(num_bases)
            order = self.orders[i-1]
            knot_sel = ((i-1,) + (0,)*(i-1) + (slice(None,None),) + 
                (0,)*(self.ndim-i))
            ts = self.knots[knot_sel]
            self.tcks.append((ts, cs, order))

    def _eval_basis(self, dim, x, nu=0):
        return process_bases_call(
            *self.tcks[dim-1][0,1], 
            x,
            np.broadcast_to(nu, (self.ndim,))[dim-1],
            self.periodic[dim-1]) 
        # interpolate.splev(x, self.tcks[dim-1], nu)

    def _eval_bases(self, x, nus=0):
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
        for i in np.arange(self.ndim)+1:
            nu = nus[i-1]
            u_mats.append(self._eval_basis(i, x[i-1,...], nu))
        return u_mats

    def __call__(self, x, nus=0):

        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.ravel(), dtype=np.float_)

        if periodic:
            n = t.size - k - 1
            x = t[k] + (x - t[k]) % (t[n] - t[k])
            ell = find_intervals(t, k, x, False)
        else:
            ell = find_intervals(t, k, x, extrapolate)

        u = eval_bases(t, k, x, ell, nu)
        u.reshape((k+1,)+x_shape)



        

        u_mats = self._eval_bases(x, nus)
        u_args = [subarg for arg in zip(u_mats, self.u_ops) for subarg in arg]
        y_out = np.einsum(self.coeffs, self.input_op, *u_args, self.output_op)
        return y_out

def make_interp_spline(x, y, bcs=0, orders=3):
    if isinstance(x, np.ndarray): # mesh
        if x.ndim == 1:
            ndim = 1
            x = x[None,...]
        else:
            ndim = x.ndim - 1

    elif not isinstance(x, str) and len(x): # vectors, or try
        ndim = len(x)
        x = np.meshgrid(x, indexing='ij')
        
    else:
        raise ValueError("Don't know how to interpret x")
    
    if y.ndim == ndim:
        # how can you tell if y needs a [None, ...] or [...] ?
        # same question as to whether ndim is y.ndim or y.ndim-1
        # I think this is the right answer. 
        y = y[None, ...]
    elif y.ndim == ndim+1:
        pass
    else:
        raise ValueError("incompatible dimension size")
    mdim = y.shape[0]
    # generally, x.shape = (ndim, n1, n2, ..., n_ndim)
    # and y.sahpe = (mdim, n1, n2, ..., n_ndim)
    bcs = np.broadcast_to(bcs, (ndim,2))
    orders = np.broadcast_to(orders, (ndim,))
    knot_shape = np.r_[x.shape]
    deriv_specs = np.asarray((bcs[:,:]>0),dtype=np.int)
    knot_shape[1:] = knot_shape[1:] + orders + 1 + deriv_specs.sum(axis=1)

    knots = np.zeros(knot_shape)
    coeffs = np.pad(y, np.r_[np.c_[0,0], deriv_specs], 'constant')

    for i in np.arange(ndim)+1:
        all_other_ax_shape = np.asarray(np.r_[coeffs.shape[1:i],
            y.shape[i+1:]], dtype=np.int)
        x_line_sel = ((i-1,) + (0,)*(i-1) + (slice(None,None),) +
            (0,)*(ndim-i))
        xp = x[x_line_sel]
        order = orders[i-1]
        for idx in np.ndindex(*all_other_ax_shape):
            offset_axes_remaining_sel = (tuple(idx[i-1:] + 
                deriv_specs[i:,0]))
            y_line_sel = ((Ellipsis,) + idx[:i-1] + 
                (slice(deriv_specs[i-1,0],-deriv_specs[i-1,0] or None),) +
                offset_axes_remaining_sel)
            coeff_line_sel = ((Ellipsis,) + idx[:i-1] + (slice(None,None),)
                + offset_axes_remaining_sel)
            line_spline = interpolate.make_interp_spline(xp,
                coeffs[y_line_sel].T,
                k = order,
                bc_type=(bc_map[(bcs[i-1,0])],
                         bc_map[(bcs[i-1,1])]),

            )
            coeffs[coeff_line_sel] = line_spline.c.T
        knots[i-1,...] = (line_spline.t[(None,)*(i-1) + 
            (slice(None),) + (None,)*(ndim-i)])
    return NDBPoly(knots, coeffs, orders, np.all(bcs==periodic, axis=1))