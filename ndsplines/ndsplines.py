import numpy as np
from scipy import interpolate

from ndsplines import _npy_bspl

try:
    from ndsplines import _bspl
except ImportError:
    default_implementation = _npy_bspl
else:
    default_implementation = _bspl

__all__ = ['pinned', 'clamped', 'extrap', 'periodic', 'BSplineNDInterpolator',
           'make_interp_spline', 'make_lsq_spline', 'default_implementation']

import operator
from scipy._lib.six import string_types
from scipy.linalg import (get_lapack_funcs, LinAlgError,
                          cholesky_banded, cho_solve_banded)

"""
TODOs:


create wrapper with callback to allow for creating anti-derivative splines, etc
(use 1D operations that can be iterated over )

break out some of the knot normalization stuff (in interpolate.make_interp_spline)
to make it easier to make lsq splines? would also be useful for making the
make_interp_spline more efficient

make sure these can be pickled (maybe store knots, coeffs, orders, etc to matfile?


"""
pinned = 2
clamped = 1
extrap = 0
periodic = -1

bc_map =  {clamped: "clamped", pinned: "natural", extrap: None, periodic: None}


class BSplineNDInterpolator(object):
    """
    Parameters
    ----------
    knots : list of ndarrays,
        shapes=[n_1+orders[i-1]+1, ..., n_ndim+orders[-1]+1], dtype=np.float_
    coefficients : ndarray, shape=(mdim, n_1, n_2, ..., n_ndim), dtype=np.float_
    orders : ndarray, shape=(ndim,), dtype=np.int_
    periodic : ndarray, shape=(ndim,), dtype=np.bool_
    extrapolate : ndarray, shape=(ndim,2), dtype=np.bool_
    """
    def __init__(self, knots, coefficients, orders, periodic=False, extrapolate=True, implementation=default_implementation):
        self.implementation = implementation
        self.knots = knots
        self.coefficients = coefficients
        self.ndim = len(knots) # dimension of knots
        self.nis = np.array([len(knot) for knot in knots])
        self.mdim = coefficients.shape[0] # dimension of coefficeints
        self.orders = np.broadcast_to(orders, (self.ndim,))
        self.periodic = np.broadcast_to(periodic, (self.ndim,))
        self.extrapolate = np.broadcast_to(extrapolate, (self.ndim,2))

        self.coefficient_op = [0,] + list(i for i in range(2,self.ndim+2)) + [1,]
        self.u_ops = [[1, i+2] for i in range(self.ndim)]
        self.output_op = [0,1]

        self.coefficient_selector_base = np.meshgrid(*[np.arange(order+1) for order in self.orders], indexing='ij')
        self.coefficient_shape_base = (self.ndim,)+tuple(self.orders+1)

        self.current_max_num_points = 0
        self.allocate_workspace_arrays(1)

        self.u_arg = [subarg for arg in zip(self.basis_workspace, self.u_ops) for subarg in arg]

    def compute_basis_coefficient_selector(self, x, nus=0):
        """
        Parameters
        ----------
        x : ndarray, shape=(self.ndim, s) dtype=np.float_
        nus : int or ndarray, shape=(self.ndim,) dtype=np.int_
        """
        num_points = x.shape[-1]

        if not isinstance(nus, np.ndarray):
            nu = nus

        for i in range(self.ndim):
            t = self.knots[i]
            k = self.orders[i]
            if isinstance(nus, np.ndarray):
                nu = nus[i]
            if self.periodic[i]:
                n = t.size - k - 1
                x[i,:] = t[k] + (x[i,:] - t[k]) % (t[n] - t[k])
                extrapolate_flag = False
            else:
                if not self.extrapolate[i,0]:
                    lt_sel = x[i,:] < t[k]
                    x[i,lt_sel] = t[k]
                if not self.extrapolate[i,1]:
                    gte_sel = t[-k-1] < x[i,:]
                    x[i,gte_sel] = t[-k-1]
                extrapolate_flag = True


            self.implementation.evaluate_spline(t, k, x[i,:], nu, extrapolate_flag, self.interval_workspace[i], self.basis_workspace[i],)
            np.add(
                self.coefficient_selector_base[i][..., None],
                self.interval_workspace[i][:num_points], 
                out=self.coefficient_selector[i, ..., :num_points])

            self.u_arg[2*i] = self.basis_workspace[i][:num_points, :self.orders[i]+1]

    def allocate_workspace_arrays(self, num_points):
        if self.current_max_num_points < num_points:
            self.current_max_num_points = num_points
            self.basis_workspace = np.empty((
                self.ndim, 
                self.current_max_num_points,
                2*np.max(self.orders)+3
            ), dtype=np.float_)
            self.interval_workspace = np.empty((self.ndim, self.current_max_num_points, ), dtype=np.intc)
            self.coefficient_selector = np.empty(self.coefficient_shape_base + (self.current_max_num_points,), dtype=np.int_)

    def __call__(self, x, nus=0):
        """
        Parameters
        ----------
        x : ndarray, shape=(self.ndim, ...) dtype=np.float_
            Point(s) to evaluate spline on. Output will be (self.mdim,...)
        nus : ndarray, broadcastable to shape=(self.ndim,) dtype=np.int_
            Order of derivative(s) for each dimension to evaluate
            
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = np.ascontiguousarray(x.reshape((self.ndim, -1)), dtype=np.float_)
        num_points = x.shape[-1]

        if isinstance(nus, np.ndarray):
            if nus.ndim != 1 or nus.size != self.ndim:
                raise ValueError("nus is wrong shape")

        self.allocate_workspace_arrays(x.shape[-1])
        self.compute_basis_coefficient_selector(x, nus)        
        coefficient_selector = (slice(None),) + tuple(self.coefficient_selector[..., :num_points])

        y_out = np.einsum(self.coefficients[coefficient_selector], self.coefficient_op, 
            *self.u_arg, 
            self.output_op)
        if (x_ndim > 1) and ((self.ndim > 1) or (size[0] != self.ndim)):
            y_out = y_out.reshape((self.mdim,) + x_shape[1:])
        return y_out

def make_lsq_spline(x, y, knots, orders, w=None, check_finite=True):
    """
    Construct an interpolating B-spline.

    Parameters
    ----------
    x : array_like, shape (ndim, num_points)
        Abscissas.
    y : array_like, shape (mdim, num_points)
        Ordinates.
    knots : iterable of array_like, shape (n_1 + orders[0] + 1,), ... (n_ndim, + orders[-1] + 1)
        Knots and data points must satisfy Schoenberg-Whitney conditions.
    orders : ndarray, shape=(ndim,), dtype=np.int_
    w : array_like, shape (num_points,), optional
        Weights for spline fitting. Must be positive. If ``None``,
        then weights are all equal.
        Default is ``None``.

    Notes
    -----
    I construct the observation matrix A, so that A@c = y
    I am not being particularly careful about structure, sparcity, etc. because 
    I am assuming a small number of knots relative to the number of data points 
    and sufficient speed from the numpy linear algebra library (i.e., MKL) to
    make it unnoticeable.


    """
    ndim = x.shape[0]
    mdim = y.shape[0]
    num_points = x.shape[1]
    assert x.shape[1] == y.shape[1]
    assert x.ndim == 2
    assert y.ndim == 2

    # TODO: do appropriate shape checks, etc.
    # TODO: check knot shape and order

    knot_shapes = tuple(knot.size - order - 1 for knot, order in zip(knots, orders))
    
    temp_spline = BSplineNDInterpolator(knots, np.empty(mdim), orders)
    temp_spline.allocate_workspace_arrays(num_points)
    temp_spline.compute_basis_coefficient_selector(x)

    observation_tensor_values = np.einsum(*temp_spline.u_arg, temp_spline.coefficient_op[1:-1] + [1,])
    observation_tensor = np.zeros((num_points,) + knot_shapes)
    observation_tensor[(np.arange(num_points),) + tuple(temp_spline.coefficient_selector[..., :num_points])] = observation_tensor_values

    observation_matrix = observation_tensor.reshape((num_points, -1))

    # TODO: implemnet weighting matrix, which I think is just matrix multiply by diag(w) on left for both observation matrix and output.

    lsq_coefficients, lsq_residuals, rank, singular_values = np.linalg.lstsq(observation_matrix, y.T)
    temp_spline.coefficients = lsq_coefficients.T.reshape((mdim,) + knot_shapes )

    # TODO: I think people will want this matrix, is there a better way to give this to a user?
    temp_spline.observation_matrix = observation_matrix
    return temp_spline


from scipy.interpolate._bsplines import _as_float_array, _get_dtype, _augknt, _convert_string_aliases, _process_deriv_spec, _bspl, prod, _not_a_knot


def make_interp_spline(x, y, bcs=0, orders=3):
    """
    Construct an interpolating B-spline.

    Parameters
    ----------
    x : array_like, shape (ndim, n_1, n_2, ..., n_ndim) or arguments to np.meshgrid to construct same
        Abscissas.
    y : array_like, shape (mdim, n_1, n_2, ..., n_ndim)
        Ordinates.
    bcs : (list of) 2-tuples or None
    orders : ndarray, shape=(ndim,), dtype=np.int_
        Degree of interpolant for each axis (or broadcastable)
    periodic : ndarray, shape=(ndim,), dtype=np.bool_
    extrapolate : ndarray, shape=(ndim,), dtype=np.bool_

    Notes
    -----
    TODO: use scipy source to implement this more efficiently?
    i.e., do knot computation once for each dimension, then coefficients for
    each line 
    """
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
    deriv_specs = np.asarray((bcs[:,:]>0),dtype=np.int)

    knots = []
    coefficients = np.pad(y, np.r_[np.c_[0,0], deriv_specs], 'constant')

    axis=0
    check_finite=True

    for i in np.arange(ndim)+1:
        all_other_ax_shape = np.asarray(np.r_[coefficients.shape[1:i],
            y.shape[i+1:]], dtype=np.int)
        x_line_sel = ((i-1,) + (0,)*(i-1) + (slice(None,None),) +
            (0,)*(ndim-i))
        x_slice = x[x_line_sel]
        k = orders[i-1]

        bc_type=(bc_map[(bcs[i-1,0])],
                 bc_map[(bcs[i-1,1])])

        #=======================================
        # START FROM SCIPY
        #=======================================
        if bc_type is None or bc_type == 'not-a-knot':
            deriv_l, deriv_r = None, None
        elif isinstance(bc_type, string_types):
            deriv_l, deriv_r = bc_type, bc_type
        else:
            try:
                deriv_l, deriv_r = bc_type
            except TypeError:
                raise ValueError("Unknown boundary condition: %s" % bc_type)
        #=======================================
        # END FROM SCIPY
        #=======================================

        for idx in np.ndindex(*all_other_ax_shape):
            offset_axes_remaining_sel = (tuple(idx[i-1:] + 
                deriv_specs[i:,0]))
            y_line_sel = ((Ellipsis,) + idx[:i-1] + 
                (slice(deriv_specs[i-1,0],-deriv_specs[i-1,0] or None),) +
                offset_axes_remaining_sel)
            coeff_line_sel = ((Ellipsis,) + idx[:i-1] + (slice(None,None),)
                + offset_axes_remaining_sel)

            y_slice = coefficients[y_line_sel].T

            t = None

            #=======================================
            # START FROM SCIPY
            #=======================================

            if not -y_slice.ndim <= axis < y_slice.ndim:
                raise ValueError("axis {} is out of bounds".format(axis))
            if axis < 0:
                axis += y_slice.ndim

            # special-case k=0 right away
            if k == 0:
                if any(_ is not None for _ in (t, deriv_l, deriv_r)):
                    raise ValueError("Too much info for k=0: t and bc_type can only "
                                     "be None.")
                x_slice = _as_float_array(x_slice, check_finite)
                t = np.r_[x_slice, x_slice[-1]]
                c = np.asarray(y_slice)
                c = np.rollaxis(c, axis)
                c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
                return BSpline.construct_fast(t, c, k, axis=axis)

            # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
            if k == 1 and t is None:
                if not (deriv_l is None and deriv_r is None):
                    raise ValueError("Too much info for k=1: bc_type can only be None.")
                x_slice = _as_float_array(x_slice, check_finite)
                t = np.r_[x_slice[0], x_slice, x_slice[-1]]
                c = np.asarray(y_slice)
                c = np.rollaxis(c, axis)
                c = np.ascontiguousarray(c, dtype=_get_dtype(c.dtype))
                return BSpline.construct_fast(t, c, k, axis=axis)

            x_slice = _as_float_array(x_slice, check_finite)
            y_slice = _as_float_array(y_slice, check_finite)
            k = operator.index(k)

            # come up with a sensible knot vector, if needed
            if t is None:
                if deriv_l is None and deriv_r is None:
                    if k == 2:
                        # OK, it's a bit ad hoc: Greville sites + omit
                        # 2nd and 2nd-to-last points, a la not-a-knot
                        t = (x_slice[1:] + x_slice[:-1]) / 2.
                        t = np.r_[(x_slice[0],)*(k+1),
                                   t[1:-1],
                                   (x_slice[-1],)*(k+1)]
                    else:
                        t = _not_a_knot(x_slice, k)
                else:
                    t = _augknt(x_slice, k)

            t = _as_float_array(t, check_finite)

            y_slice = np.rollaxis(y_slice, axis)    # now internally interp axis is zero

            if x_slice.ndim != 1 or np.any(x_slice[1:] <= x_slice[:-1]):
                raise ValueError("Expect x_slice to be a 1-D sorted array_like.")
            if k < 0:
                raise ValueError("Expect non-negative k.")
            if t.ndim != 1 or np.any(t[1:] < t[:-1]):
                raise ValueError("Expect t to be a 1-D sorted array_like.")
            if x_slice.size != y_slice.shape[0]:
                raise ValueError('x_slice and y_slice are incompatible.')
            if t.size < x_slice.size + k + 1:
                raise ValueError('Got %d knots, need at least %d.' %
                                 (t.size, x_slice.size + k + 1))
            if (x_slice[0] < t[k]) or (x_slice[-1] > t[-k]):
                raise ValueError('Out of bounds w/ x_slice = %s.' % x_slice)

            # Here : deriv_l, r = [(nu, value), ...]
            deriv_l = _convert_string_aliases(deriv_l, y_slice.shape[1:])
            deriv_l_ords, deriv_l_vals = _process_deriv_spec(deriv_l)
            nleft = deriv_l_ords.shape[0]

            deriv_r = _convert_string_aliases(deriv_r, y_slice.shape[1:])
            deriv_r_ords, deriv_r_vals = _process_deriv_spec(deriv_r)
            nright = deriv_r_ords.shape[0]

            # have `n` conditions for `nt` coefficients; need nt-n derivatives
            n = x_slice.size
            nt = t.size - k - 1

            if nt - n != nleft + nright:
                raise ValueError("The number of derivatives at boundaries does not "
                                 "match: expected %s, got %s+%s" % (nt-n, nleft, nright))

            # set up the LHS: the collocation matrix + derivatives at boundaries
            kl = ku = k
            ab = np.zeros((2*kl + ku + 1, nt), dtype=np.float_, order='F')
            _bspl._colloc(x_slice, t, k, ab, offset=nleft)
            if nleft > 0:
                _bspl._handle_lhs_derivatives(t, k, x_slice[0], ab, kl, ku, deriv_l_ords)
            if nright > 0:
                _bspl._handle_lhs_derivatives(t, k, x_slice[-1], ab, kl, ku, deriv_r_ords,
                                        offset=nt-nright)

            # set up the RHS: values to interpolate (+ derivative values, if any)
            extradim = prod(y_slice.shape[1:])
            rhs = np.empty((nt, extradim), dtype=y_slice.dtype)
            if nleft > 0:
                rhs[:nleft] = deriv_l_vals.reshape(-1, extradim)
            rhs[nleft:nt - nright] = y_slice.reshape(-1, extradim)
            if nright > 0:
                rhs[nt - nright:] = deriv_r_vals.reshape(-1, extradim)

            # solve Ab @ x_slice = rhs; this is the relevant part of linalg.solve_banded
            if check_finite:
                ab, rhs = map(np.asarray_chkfinite, (ab, rhs))
            gbsv, = get_lapack_funcs(('gbsv',), (ab, rhs))
            lu, piv, c, info = gbsv(kl, ku, ab, rhs,
                    overwrite_ab=True, overwrite_b=True)

            if info > 0:
                raise LinAlgError("Collocation matix is singular.")
            elif info < 0:
                raise ValueError('illegal value in %d-th argument of internal gbsv' % -info)

            c = np.ascontiguousarray(c.reshape((nt,) + y_slice.shape[1:]))

            #=======================================
            # START FROM SCIPY
            #=======================================

            coefficients[coeff_line_sel] = c.T
        knots.append(t)
    return BSplineNDInterpolator(knots, coefficients, orders, 
        np.all(bcs==periodic, axis=1),
        (bcs >= 0))
