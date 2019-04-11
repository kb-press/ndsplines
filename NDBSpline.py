import numpy as np
from scipy import interpolate
import scipy_bspl

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


class NDBSpline(object):
    def __init__(self, knots, coefficients, orders, periodic=False, extrapolate=True):
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
        self.knots = knots
        self.coefficients = coefficients
        self.ndim = len(knots) # dimension of knots
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


            scipy_bspl.evaluate_spline(t, k, x[i,:], nu, extrapolate_flag, self.interval_workspace[i], self.basis_workspace[i],)
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
            self.interval_workspace = np.empty((self.ndim, self.current_max_num_points, ), dtype=np.int_)
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
            if nus.shape != 1 and nus.shape != self.ndim:
                raise ValueError("nus is wrong shape")

        self.allocate_workspace_arrays(x.shape[-1])
        self.compute_basis_coefficient_selector(x, nus)        
        coefficient_selector = (slice(None),) + tuple(self.coefficient_selector[..., :num_points])

        y_out = np.einsum(self.coefficients[coefficient_selector], self.coefficient_op, 
            *self.u_arg, 
            self.output_op)
        y_out = y_out.reshape(
                    (self.mdim,) + x_shape[1:] if x_ndim!=1 else x_shape)
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
    w : array_like, shape (n,), optional
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
    
    temp_spline = NDBSpline(knots, np.empty(mdim), orders)
    temp_spline.allocate_workspace_arrays(num_points)
    temp_spline.compute_basis_coefficient_selector(x)

    observation_tensor_values = np.einsum(*temp_spline.u_arg, temp_spline.coefficient_op[1:-1] + [1,])
    observation_tensor = np.zeros((num_points,) + knot_shapes)
    observation_tensor[(np.arange(num_points),) + tuple(temp_spline.coefficient_selector[..., :num_points])] = observation_tensor_values

    observation_matrix = observation_tensor.reshape((num_points, -1))

    # TODO: implemnet weighting matrix, which I think is just matrix multiply by diag(w)?

    lsq_coefficients, lsq_residuals, rank, singular_values = np.linalg.lstsq(observation_matrix, y.T)
    temp_spline.coefficients = lsq_coefficients.T.reshape((mdim,) + knot_shapes )

    # TODO: I think people will want this matrix, is there a better way to give this to a user?
    temp_spline.observation_matrix = observation_matrix
    return temp_spline


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

    for i in np.arange(ndim)+1:
        all_other_ax_shape = np.asarray(np.r_[coefficients.shape[1:i],
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
                coefficients[y_line_sel].T,
                k = order,
                bc_type=(bc_map[(bcs[i-1,0])],
                         bc_map[(bcs[i-1,1])]),

            )
            coefficients[coeff_line_sel] = line_spline.c.T
        knots.append(line_spline.t)
    return NDBSpline(knots, coefficients, orders, 
        np.all(bcs==periodic, axis=1),
        (bcs %2)==0)