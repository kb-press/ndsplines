import numpy as np
import operator
from scipy.linalg import get_lapack_funcs, LinAlgError
from scipy.interpolate._bsplines import (prod, _as_float_array,
                                         _bspl as _sci_bspl)

from ndsplines import _npy_bspl

__all__ = ['impl', 'NDSpline', '_not_a_knot', 'make_interp_spline', 
           'make_lsq_spline', 'make_interp_spline_from_tidy', 'from_file']

impl = _npy_bspl


class NDSpline(object):
    r"""Multivariate tensor-product spline in the B-spline basis. A general `N`
    dimensional tensor product B-spline is given by 

    .. math::

        S(x_1, ..., x_N) = \sum_{i_1=0}^{n_1-1}  \cdots \sum_{i_N=0}^{n_N-1} c_{i_1, ..., i_N} \prod_{j = i}^{N} B_{i_j;k_j,t_j}(x_j)

    where :math:`B_{i_j;k_j,t_j}` is the :math:`i_j`-th B-spline basis function
    of degree :math:`k_j` over the knots :math:`{t_j}` with coefficients
    :math:`c_{i_1, ..., i_N}`

    Parameters
    ----------
    knots : list of ndarrays, shapes=[n_0+degrees[i]+1, ..., n_ndim+degrees[-1]+1], dtype=np.float_
        List of knots in each dimension, :math:`[t_1, t_2, \ldots, t_N]`.
    coefficients : ndarray, shape=(n_1, n_2, ..., n_xdim) + yshape, dtype=np.float_
        N-D array of coefficients, :math:`c_{i_1, ..., i_N}`.
    degrees : ndarray, shape=(xdim,), dtype=np.int_
        Array of the degree of each dimension, :math:`[k_1, k_2, \ldots, k_N]`.
    periodic : ndarray, shape=(xdim,), dtype=np.bool_
        Array of periodicity flags for each dimension.
    extrapolate : ndarray, shape=(xdim,2), dtype=np.bool_
        Array of extrapolation flags for each side in each dimension.

    Attributes
    ----------
    knots : list of ndarrays, shapes=[n_0+degrees[i]+1, ..., n_ndim+degrees[-1]+1], dtype=np.float_
        List of knots in each dimension, :math:`[t_1, t_2, \ldots, t_N]`.
    xdim : int
        Dimension of spline input space.
    coefficients : ndarray, shape=(n_1, n_2, ..., n_xdim, ydim), dtype=np.float_
        N-D array of coefficients, :math:`c_{i_1, ..., i_N}`.
    ydim : int
        Dimension of spline output space.
    yshape : tuple of ints
        Shape of spline output.

    """

    def __init__(self, knots, coefficients, degrees, periodic=False, extrapolate=True):
        self.knots = knots
        self.xdim = len(knots) # dimension of knots
        self.xshape = tuple(knot.size for knot in knots)
        
        self.degrees = np.broadcast_to(degrees, (self.xdim,))
        self.max_degree = np.max(degrees)
        self.periodic = periodic
        self.extrapolate = extrapolate

        expected_shape = self.xshape - self.degrees - 1
        if not np.all(coefficients.shape[:self.xdim] == expected_shape):
            raise ValueError("Expected coefficients.shape to start with %s, got %s." % (repr(expected_shape), repr(coefficients.shape[:self.xdim])))

        self.yshape = coefficients.shape[self.xdim:]
        self.ydim = prod(self.yshape)
        self.coefficients = coefficients.reshape( tuple(self.xshape - self.degrees - 1,) + (self.ydim,))

        self.coefficient_op = list(i for i in range(self.xdim+2))
        self.u_ops = [[self.xdim, i] for i in range(self.xdim)]
        self.output_op = [self.xdim, self.xdim+1]

        self.coefficient_selector_base = np.meshgrid(*[np.arange(degree+1) for degree in self.degrees], indexing='ij')
        self.coefficient_shape_base = tuple(self.degrees+1) + (self.xdim,)

        self.current_max_num_points = 0
        self.allocate_workspace_arrays(1)

        self.u_arg = [subarg for arg in zip(self.basis_workspace, self.u_ops) for subarg in arg]

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, periodic):
        self._periodic = np.broadcast_to(periodic, (self.xdim,))

    @property
    def extrapolate(self):
        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, extrapolate):
        if not isinstance(extrapolate, np.ndarray) and not np.isscalar(extrapolate):
            extrapolate = np.array(extrapolate)
        if isinstance(extrapolate, np.ndarray):
            if extrapolate.ndim == 1:
                extrapolate = extrapolate[:,None]

        self._extrapolate = np.broadcast_to(extrapolate, (self.xdim,2))

    def allocate_workspace_arrays(self, num_points):
        """
        Allocate workspace arrays for the N-dimensional B-spline evaluation.

        Parameters
        ----------
        num_points : int
            Number of evaluation points to allocate for.

        See Also
        --------
        NDSpline.__call__, NDSpline.allocate_workspace_arrays
        """
        if self.current_max_num_points < num_points:
            self.current_max_num_points = num_points
            self.basis_workspace = np.empty((
                self.xdim,
                self.current_max_num_points,
                2*self.max_degree+2,
            ), dtype=np.float_)
            self.interval_workspace = np.empty((self.xdim, self.current_max_num_points, ), dtype=np.intc)
            self.coefficient_selector = np.empty((self.current_max_num_points,) + self.coefficient_shape_base, dtype=np.intc)

    def compute_basis_coefficient_selector(self, x, nus=0):
        """
        Evaluate the N-dimensional B-spline basis functions and coefficient 
        selectors. This is an intermediate computation for evaluating the 
        B-spline.

        Parameters
        ----------
        x : ndarray, shape=(s, self.xdim,) dtype=np.float_
            Points at which to evaluate the spline.
        nus : int or ndarray, shape=(self.xdim,) dtype=np.int_
            Order of derivatives for each dimension to evaluate. Optional, 
            default is 0.

        See Also
        --------
        NDSpline.__call__, NDSpline.allocate_workspace_arrays
        """
        num_points = x.shape[0]

        if not isinstance(nus, np.ndarray):
            nu = nus

        for i in range(self.xdim):
            t = self.knots[i]
            k = self.degrees[i]
            if isinstance(nus, np.ndarray):
                nu = nus[i]
            if self.periodic[i]:
                n = t.size - k - 1
                x[:,i] = t[k] + (x[:,i] - t[k]) % (t[n] - t[k])
                extrapolate_flag = False
            else:
                if not self.extrapolate[i,0]:
                    lt_sel = x[:, i] < t[k]
                    x[lt_sel, i] = t[k]
                if not self.extrapolate[i,1]:
                    gte_sel = t[-k-1] < x[:, i]
                    x[gte_sel, i] = t[-k-1]
                extrapolate_flag = True


            impl.evaluate_spline(t, k, x[:,i], nu, extrapolate_flag, self.interval_workspace[i], self.basis_workspace[i],)
            np.add(
                self.coefficient_selector_base[i][None, ...],
                # Broadcasting does not play nciely with xdim as last axis for some reason,
                # so broadcasting manually. Need to determine speed concequences.
                self.interval_workspace[i][(slice(0,num_points),) + (None,)*self.xdim],
                out=self.coefficient_selector[:num_points, ..., i])

            self.u_arg[2*i] = self.basis_workspace[i, :num_points, :self.degrees[i]+1]

    def __call__(self, x, nus=0):
        """
        Evaluate the N-dimensional B-spline.

        Parameters
        ----------
        x : ndarray, shape=(..., self.xdim) dtype=np.float_
            Points at which to evaluate the spline. 
        nus : ndarray, broadcastable to shape=(self.xdim,) dtype=np.int_
            Order of derivatives for each dimension to evaluate. Optional, 
            default is 0.

        Returns
        -------
        y : ndarray, shape=(..., self.yshape) dtype=np.float_
            Value of N-dimensional B-spline at x.

        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        else:
            x = x.copy(order='C')

        if x.ndim == 1 and self.xdim==1: # multiple points in 1D space
            x = x[:, None]
        elif x.ndim == 1 and x.size == self.xdim: # 1 point in ND space
            x = x[None, :]

        x_shape, x_ndim = x.shape, x.ndim
        # need to double transpose so slices of all `i`th dim coords are c-contiguous
        x = np.ascontiguousarray(x.reshape((-1, self.xdim)).T, dtype=np.float_).T
        num_points = x.shape[0]

        if isinstance(nus, np.ndarray):
            if nus.ndim != 1 or nus.size != self.xdim:
                raise ValueError("nus is wrong shape")

        self.allocate_workspace_arrays(x.shape[0])
        self.compute_basis_coefficient_selector(x, nus)
        coefficient_selector = tuple(self.coefficient_selector[:num_points, ...].swapaxes(0,-1)) + (slice(None),)

        y_out = np.einsum(self.coefficients[coefficient_selector], self.coefficient_op,
            *self.u_arg,
            self.output_op)

        return y_out.reshape((x_shape[:-1] + self.yshape))

    def derivative(self, dim, nu=1):
        """
        Return `NDSpline` representing the `nu`-th derivative in the `dim`-th
        dimension.

        Parameters
        ----------
        dim : int
            Dimension in which to take the derivative. 0-indexed, so 
            must be  >= 0 and < self.xdim.
        nu : int, optional
            Derivative order, default is 1.

        Returns
        -------
        b : NDSpline object
            A new instance representing the derivative.

        See Also
        --------
        NDSpline.antiderivative
        """
        if nu < 0:
            return self.antiderivative(dim, nu=-nu)

        k = self.degrees[dim]

        if nu > k:
            raise ValueError("Order of derivative nu = %d must be <= degree of spline %d for the %d-th dimension" % (nu, k, dim))

        coefficients = self.coefficients.copy()
        degrees = self.degrees.copy()
        knots = [knot.copy() for knot in self.knots]
        t = knots[dim]
        c_left_selector = [slice(None)]*self.xdim + [...,]
        c_right_selector = [slice(None)]*self.xdim + [...,]
        dt_shape = (None,)*dim + (slice(None),) + (None,)*(self.xdim-dim)

        with np.errstate(invalid='raise', divide='raise'):
            try:
                for j in range(nu):
                    # See e.g. Schumaker, Spline Functions: Basic Theory, Chapter 5

                    # Compute the denominator in the differentiation formula.
                    # (and append traling dims, if necessary)
                    dt = (t[k+1:-1] - t[1:-k-1])[dt_shape]
                    # Compute the new coefficients
                    c_left_selector[dim] = slice(1,None)
                    c_right_selector[dim] = slice(0,-1)
                    coefficients = (coefficients[tuple(c_left_selector)] -
                                    coefficients[tuple(c_right_selector)]) * k / dt
                    # Adjust knots
                    t = t[1:-1]
                    k -= 1
            except FloatingPointError:
                raise ValueError(("The spline has internal repeated knots "
                                  "and is not differentiable %d times") % n)
        degrees[dim] = k
        knots[dim] = t

        return self.__class__(knots, coefficients.reshape(coefficients.shape[:self.xdim] + self.yshape), degrees, self.periodic, self.extrapolate)

    def antiderivative(self, dim, nu=1):
        """
        Return `NDSpline` representing the `nu`-th antiderivative in the 
        `dim`-th dimension.

        Parameters
        ----------
        dim : int
            Dimension in which to take the antiderivative. 0-indexed, so 
            must be  >= 0 and < self.xdim.
        nu : int, optional
            Antiderivative order. Default is 1.

        Returns
        -------
        b : NDSpline object
            A new instance representing the antiderivative.

        See Also
        --------
        NDSpline.derivative
        """
        if nu < 0:
            return self.derivative(dim, nu=-nu)
            
        k = self.degrees[dim]
        coefficients = self.coefficients.copy()
        degrees = self.degrees.copy()
        knots = [knot.copy() for knot in self.knots]
        t = knots[dim]
        dt_shape = (None,)*dim + (slice(None),) + (None,)*(self.xdim-dim)
        left_pad_width = [(0,0)]*dim + [(1,0)] + [(0,0)]*(self.xdim-dim)
        for j in range(nu):
            dt = (t[k+1:] - t[:-k-1])[dt_shape]
            # Compute the new coefficients
            coefficients = np.cumsum(coefficients * dt, axis=dim) / (k+1)
            coefficients = np.pad(coefficients, left_pad_width, 'constant')
            # Adjust knots
            t = np.r_[t[0], t, t[-1]] 
            k += 1
            # right_pad_width[dim] = (0,k)

        degrees[dim] = k
        knots[dim] = t

        return self.__class__(knots, coefficients.reshape(coefficients.shape[:self.xdim] + self.yshape), degrees, self.periodic, self.extrapolate)


    def to_file(self, file, compress=True):
        """
        Save attributes of ``NDSpline`` object to binary file in NumPy ``.npz`` 
        format so that the object can be re-created.

        Parameters
        ----------
        file : file, str, or pathlib.Path
            File or filename to which the data is saved.  If file is a file-object,
            then the filename is unchanged.  If file is a string or Path, a ``.npz``
            extension will be appended to the file name if it does not already
            have one.
        compress : bool, optional
            Whether to compress the archive of attributes.  Optional, 
            default is True.

        Notes
        -----
        Saves knots in order with name "knots_%d" where %d is the dimension
        of the input space.

        See Also
        --------
        from_file

        """
        to_save_dict = {}
        for idx, knot in zip(range(self.xdim), self.knots):
            to_save_dict['knots_%d' % idx] = knot
        to_save_dict['coefficients'] = self.coefficients.reshape(self.coefficients.shape[:self.xdim] + self.yshape)
        to_save_dict['degrees'] = self.degrees
        to_save_dict['periodic'] = self.periodic
        to_save_dict['extrapolate'] = self.extrapolate

        if compress:
            np.savez_compressed(file, **to_save_dict)
        else:
            np.savez(file, **to_save_dict)

    def copy(self):
        """
        Return a deep copy of this `NDSpline` object.

        Returns
        -------
        b : NDSpline object
            A deep copy of this spline.
        """
        return self.__class__(
        [knot.copy() for knot in self.knots],
        self.coefficients.copy(),
        self.degrees.copy(),
        self.periodic.copy(),
        self.extrapolate.copy(),
        )

    def __eq__(self, other):
        """
        Check equality with another spline.

        Parameters
        ----------
        other : NDSpline object
            Other spline to check equality with.
        """
        status = True
        for knot_left, knot_right in zip(self.knots, other.knots):
            status = status & np.allclose(knot_left, knot_right)

        status = status & np.allclose(self.coefficients, other.coefficients)

        status = status & np.all(self.degrees == other.degrees)
        status = status & np.all(self.periodic == other.periodic)
        status = status & np.all(self.extrapolate == other.extrapolate)
        status = status & (self.yshape == other.yshape)
        return status



def from_file(file):
    """
    Create a `NDSpline` object from a NumPy archive containing the
    necessary attributes.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the
        ``seek()`` and ``read()`` methods. Pickled files require that the
        file-like object support the ``readline()`` method as well.

    Notes
    -----
    Assumes knots are saved in order with names "knots_%d" where %d is 
    the dimension of the input space.

    See Also
    --------
    NDSpline.to_file
    """
    with np.load(file) as data:
        coefficients = data['coefficients']
        degrees = data['degrees']
        periodic = data['periodic']
        extrapolate = data['extrapolate']
        knots = [ data[key] for key in data.keys() if key.startswith("knots_") ]
    return NDSpline(knots, coefficients, degrees, periodic, extrapolate)


def make_lsq_spline(x, y, knots, degrees, w=None, check_finite=True):
    """
    Construct a least squares regression B-spline.

    Parameters
    ----------
    x : array_like, shape (num_points, xdim)
        Abscissas.
    y : array_like, shape (num_points, ydim)
        Ordinates.
    knots : iterable of array_like, shape (n_1 + degrees[0] + 1,), ... (n_xdim, + degrees[-1] + 1)
        Knots and data points must satisfy Schoenberg-Whitney conditions.
    degrees : ndarray, shape=(xdim,), dtype=np.int_                           
    w : array_like, shape (num_points,), optional
        Weights for spline fitting. Must be positive. If ``None``,
        then weights are all equal. Default is ``None``.

    Returns
    -------
    b : NDSpline object
        A least-squares `NDSpline`.

    See Also
    --------
    make_interp_spline, make_interp_spline_from_tidy

    """
    if x.ndim == 1:
        x = x[:, None]
    xdim = x.shape[1]
    num_points = x.shape[0]

    yshape = y.shape[1:]
    ydim = prod(yshape)
    y = y.reshape(num_points, ydim)

    # make slices c-contiguous
    x = x.T.copy(order='C').T
    knot_shapes = tuple(knot.size - degree - 1 for knot, degree in zip(knots, degrees))

    temp_spline = NDSpline(knots, np.empty(knot_shapes + yshape), degrees)
    temp_spline.allocate_workspace_arrays(num_points)
    temp_spline.compute_basis_coefficient_selector(x)

    observation_tensor_values = np.einsum(*temp_spline.u_arg, temp_spline.coefficient_op[:-1])
    observation_tensor = np.zeros((num_points,) + knot_shapes)
    observation_tensor[(np.arange(num_points),) + tuple(temp_spline.coefficient_selector[:num_points, ...].swapaxes(0,-1))] = observation_tensor_values

    observation_matrix = observation_tensor.reshape((num_points, -1))

    if w is not None:
        y = w[:, None]*y
        observation_matrix = w[:, None] * observation_matrix

    lsq_coefficients, lsq_residuals, rank, singular_values = np.linalg.lstsq(observation_matrix, y, rcond=None)

    temp_spline.coefficients = lsq_coefficients.reshape(knot_shapes + yshape)
    temp_spline = NDSpline(knots, lsq_coefficients.reshape(knot_shapes + yshape), degrees)

    # TODO: I think people will want this matrix, is there a better way to give this to a user?
    temp_spline.observation_matrix = observation_matrix
    return temp_spline

def _not_a_knot(x, k, left=True, right=True):
    """Utility function to perform the knot portion of the not-a-knot procedure.

    Parameters
    ----------
    x : ndarray, shape=(n,), dtype=np.float_
        Knot array to perform the not-a-knot procedure on
    k : int
        Degree of desired spline
    left : bool
        Whether to apply not-a-knot to the left side. Optional, default is True.
    right : bool
        Whether apply to not-a-knot to the right side. Optional, default is
        True.

    Returns
    -------
    t : ndarray
        Knot array with left and/or right not-a-knot procedure applied.
    """
    if k > 2 and k % 2 == 0:
        raise ValueError("Odd degree for now only. Got %s." % k)
    if k==0: # special case for k = 0, only apply to right
        left = False
    t = x
    if left:
        t = np.r_[(t[0],)*(k+1), t[(k -1) //2 +1:]]
    if right:
        t = np.r_[t[:-(k-1)//2 -1 or None], (t[-1],)*(k+1)]
    return t

def make_interp_spline(x, y, degrees=3):
    """Construct an interpolating B-spline.

    Parameters
    ----------
    x : array_like broadcastable to (n_0, n_1, ..., n_(xdim-1), xdim) or arguments to ``numpy.meshgrid`` to construct same.
        Abscissas.
    y : array_like, shape (n_0, n_1, ..., n_(xdim-1),) + yshape
        Ordinates.
    degrees : ndarray, shape=(xdim,), dtype=np.intc
        Degree of interpolant for each axis (or broadcastable). Optional, 
        default is 3.

    Returns
    -------
    b : NDSpline object
        An interpolating `NDSpline`.

    See Also
    --------
    make_lsq_spline, make_interp_spline_from_tidy

    """
    if isinstance(x, np.ndarray):  # mesh
        if x.ndim == 1:
            x = x[..., None]
        xdim = x.shape[-1]
    elif not isinstance(x, str) and len(x):  # vectors
        xdim = len(x)
        x = np.stack(np.meshgrid(*x, indexing='ij'), axis=-1)
    else:
        raise ValueError("Don't know how to interpret x")
    
    if not np.all(y.shape[:xdim] == x.shape[:xdim]):
        raise ValueError("Expected y.shape to start with %s, got %s." % (repr(x.shape[:xdim]), repr(y.shape[:xdim])))

    yshape = y.shape[xdim:]
    ydim = prod(yshape)


    # generally, x.shape = (n_0, n_1, ..., n_(xdim-1), xdim)
    # and y.sahpe = (n_0, n_1, ..., n_(xdim-1), ydim)

    degrees = np.broadcast_to(degrees, (xdim,))

    # broadcasting does not play nicely with xdim as last axis for some reason
    bcs = np.broadcast_to(0, (xdim, 2, 2))
    deriv_specs = np.asarray((bcs[:, :, 0] > 0), dtype=np.int)
    nak_spec = np.asarray((bcs[:, :, 0] <= 0), dtype=np.bool)

    knots = []
    coefficients = np.pad(y.reshape(x.shape[:-1] + (ydim,)), np.r_[deriv_specs, np.c_[0, 0]], 'constant')

    axis = 0
    check_finite = True

    for i in np.arange(xdim):
        all_other_ax_shape = np.asarray(np.r_[coefficients.shape[:i],
            y.shape[i+1:xdim]], dtype=np.int)
        x_line_sel = ((0,)*(i) + (slice(None,None),) +
            (0,)*(xdim-i-1) + (i,))
        x_slice = x[x_line_sel]
        k = degrees[i]

        left_nak, right_nak = nak_spec[i, :]
        both_nak = left_nak and right_nak

        # Here : deriv_l, r = [(nu, value), ...]
        deriv_l_ords, deriv_r_ords = bcs[i, :, 0].astype(np.int_)

        x_slice = _as_float_array(x_slice, check_finite)
        # should there be a general check for k <= deriv_spec ?

        if k == 0:
            # all derivatives are fully defined, can only set 0-th derivative,
            # special case for nearest-neighbor, causal/anti-causal zero-order
            # hold
            if not both_nak:
                raise ValueError("Too much info for k=0: t and bc_type can only "
                                 "be notaknot.")

            left_zero, right_zero = (bcs[i, :, 1]==0)

            if left_zero and right_zero:
                t = np.r_[x_slice[0], (x_slice[:-1] + x_slice[1:])/2., x_slice[-1]]
            elif not left_zero and right_zero:
                t = np.r_[x_slice, x_slice[-1]]
            elif left_zero and not right_zero:
                t = np.r_[x_slice[0], x_slice]
            else:
                raise ValueError("Set deriv_spec = 0, with up to one side = -1 for k=0")

        # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
        if k == 1:
            # all derivatives are fully defined, can only set 0-th derivative,
            # aka not-a-knot boundary conditions to both sides
            if not both_nak:
                raise ValueError("Too much info for k=1: bc_type can only be notaknot.")

        if k==2:
            # it's possible this may be the best behavior for all even k > 0
            if both_nak:
                # special, ad-hoc case using greville sites
                t = (x_slice[1:] + x_slice[:-1]) / 2.
                t = np.r_[(x_slice[0],)*(k+1),
                           t[1:-1],
                           (x_slice[-1],)*(k+1)]

            elif left_nak or right_nak:
                raise ValueError("For k=2, can set both sides or neither side to notaknot")
            else:
                t = x_slice

        elif k != 0:
            t = _not_a_knot(x_slice, k, left_nak, right_nak)

        t = _as_float_array(t, check_finite)


        if left_nak:
            deriv_l_ords = np.array([])
            deriv_l_vals = np.array([])
            nleft = 0
        else:
            deriv_l_ords = np.array([bcs[i-1, 0, 0]], dtype=np.int_)
            deriv_l_vals = np.broadcast_to(bcs[i-1, 0, 1], ydim)
            nleft = 1

        if right_nak:
            deriv_r_ords = np.array([])
            deriv_r_vals = np.array([])
            nright = 0
        else:
            deriv_r_ords = np.array([bcs[i-1, 1, 0]], dtype=np.int_)
            deriv_r_vals = np.broadcast_to(bcs[i-1, 1, 1], ydim)
            nright = 1

        # have `n` conditions for `nt` coefficients; need nt-n derivatives
        n = x_slice.size
        nt = t.size - k - 1

        # this also catches if deriv_spec > k-1, possibly?
        if np.clip(nt - n, 0, np.inf).astype(int) != nleft + nright:
            raise ValueError("The number of derivatives at boundaries does not "
                             "match: expected %s, got %s+%s" % (nt-n, nleft, nright))

        # set up the LHS: the collocation matrix + derivatives at boundaries
        kl = ku = k
        ab = np.zeros((2*kl + ku + 1, nt), dtype=np.float_, order='F')
        _sci_bspl._colloc(x_slice, t, k, ab, offset=nleft)
        if nleft > 0:
            _sci_bspl._handle_lhs_derivatives(t, k, x_slice[0], ab, kl, ku, deriv_l_ords)
        if nright > 0:
            _sci_bspl._handle_lhs_derivatives(t, k, x_slice[-1], ab, kl, ku, deriv_r_ords,
                                    offset=nt-nright)

        knots.append(t)

        if k >= 2:
            if x_slice.ndim != 1 or np.any(x_slice[1:] <= x_slice[:-1]):
                raise ValueError("Expect x_slice to be a 1-D sorted array_like.")
            if k < 0:
                raise ValueError("Expect non-negative k.")
            if t.ndim != 1 or np.any(t[1:] < t[:-1]):
                raise ValueError("Expect t to be a 1-D sorted array_like.")
            if t.size < x_slice.size + k + 1:
                raise ValueError('Got %d knots, need at least %d.' %
                                 (t.size, x_slice.size + k + 1))
            if (x_slice[0] < t[k]) or (x_slice[-1] > t[-k]):
                raise ValueError('Out of bounds w/ x_slice = %s.' % x_slice)


        for idx in np.ndindex(*all_other_ax_shape):
            offset_axes_remaining_sel = (tuple(idx[i:] +
                deriv_specs[i+1:, 0]))
            y_line_sel = (idx[:i] +
                (slice(deriv_specs[i,0],-deriv_specs[i, 1] or None),) +
                offset_axes_remaining_sel + (Ellipsis,))
            coeff_line_sel = (idx[:i] + (slice(None,None),)
                + offset_axes_remaining_sel + (Ellipsis,))
            y_slice = coefficients[y_line_sel]

            # special-case k=0 right away
            if k == 0:
                c = np.asarray(y_slice)

            # special-case k=1 (e.g., Lyche and Morken, Eq.(2.16))
            elif k == 1:
                c = np.asarray(y_slice)

            else:
                y_slice = _as_float_array(y_slice, check_finite)
                k = operator.index(k)

                if x_slice.size != y_slice.shape[0]:
                    raise ValueError('x_slice and y_slice are incompatible.')

                # set up the RHS: values to interpolate (+ derivative values, if any)
                rhs = np.empty((nt, ydim), dtype=y_slice.dtype)
                if nleft > 0:
                    rhs[:nleft] = deriv_l_vals.reshape(-1, ydim)
                rhs[nleft:nt - nright] = y_slice.reshape(-1, ydim)
                if nright > 0:
                    rhs[nt - nright:] = deriv_r_vals.reshape(-1, ydim)

                # solve Ab @ x_slice = rhs; this is the relevant part of linalg.solve_banded
                if check_finite:
                    ab, rhs = map(np.asarray_chkfinite, (ab, rhs))
                gbsv, = get_lapack_funcs(('gbsv',), (ab, rhs))
                lu, piv, c, info = gbsv(kl, ku, ab, rhs,
                        overwrite_ab=False, overwrite_b=True)

                if info > 0:
                    raise LinAlgError("Collocation matix is singular.")
                elif info < 0:
                    raise ValueError('illegal value in %d-th argument of internal gbsv' % -info)

                c = np.ascontiguousarray(c.reshape((nt,) + y_slice.shape[1:]))

            coefficients[coeff_line_sel] = c
    coefficients = coefficients.reshape(coefficients.shape[:xdim] + yshape)
    return NDSpline(knots, coefficients, degrees,)


try:
    import pandas as pd
except ImportError:
    check_pandas = False
else:
    check_pandas = True

def make_interp_spline_from_tidy(tidy_data, input_vars, output_vars, degrees=3):
    """
    Construct an interpolating B-spline from a tidy data source. The tidy data
    source should be a complete matrix (see the ``tidy_data`` parameter 
    description). The order of the input and output dimension of the constructed
    interpolant will be the same as the ``input_vars`` and ``output_vars``
    arguments, following the convention of input variables first.

    Parameters
    ----------
    tidy_data : array_like, shape (num_points, xdim + ydim)
        Pandas DataFrame or NumPy Array of data. In order to be a complete 
        matrix, we must have
        num_points = prod( nunique(input_var) for input_var in input_vars)
        Any missing or extra data will cause an error on reshape.
    input_vars : iterable 
        Column names (for DataFrame) or indices (for np.ndarray) for input
        variables.
    output_vars : iterable 
        Column names (for DataFrame) or indices (for np.ndarray) for output
        variables.
    degrees : ndarray, shape=(ndim,), dtype=np.intc
        Degree of interpolant for each axis (or broadcastable). Optional, 
        default is 3.

    Returns
    -------
    b : NDSpline object
        An interpolating `NDSpline`.

    See Also
    --------
    make_lsq_spline, make_interp_spline
    """

    if check_pandas and isinstance(tidy_data, pd.DataFrame):
        tidy_df = tidy_data
        input_vars = [tidy_df.columns.get_loc(input_var) for input_var in input_vars]
        output_vars = [tidy_df.columns.get_loc(output_var) for output_var in output_vars]
        tidy_data = tidy_data.values
        is_pandas = True
    else:
        input_vars = list(input_vars)
        output_vars = list(output_vars)

    meshgrid_shape = [np.unique(tidy_data[:, input_var]).size for input_var in input_vars] + [tidy_data.shape[1],]

    sort_indices = np.lexsort(tidy_data[:,input_vars[::-1]].T)
    sorted_data = tidy_data[sort_indices, :]
    meshgrid_data = sorted_data.reshape(
            meshgrid_shape
        )

    xdata = meshgrid_data[..., input_vars]
    ydata = meshgrid_data[..., output_vars]
    return make_interp_spline(xdata, ydata, degrees)
