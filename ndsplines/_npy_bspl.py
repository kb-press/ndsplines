"""
NumPy implementation for evaluating B-splines.

"""
import numpy as np

def find_interval(t, k, xvals, extrapolate=False, workspace=None):
    """
    Find an interval ell such that t[ell] <= xvals < t[ell+1].

    Parameters
    ----------
    t : ndarray, shape=(n+k+1,) dtype=np.float_
        knots
    k : int
        degree of B-spline
    xvals : ndarray, shape=(s,) dtype=np.float_
        values to find the interval for
    extrapolate : bool, optional
        whether to return the last or the first interval if xval
        is out of bounds.
    workspace : ndarray, shape (s,), dtype=int or None
        Array used to return identified intervals, modified in-place.

    Returns
    -------
    ell : ndarray, shape=(s,) dtype=np.intc
        Suitable interval or -1 for each value in xvals. If workspace is provided,


    Notes
    -----
    This is a vectorized, NumPy implementation similar to the the inline 
    `_bspl.find_interval`.
    """
    if xvals.ndim != 1:
        raise ValueError("expected 1-dimensional xvals")

    s = xvals.size

    if (not isinstance(workspace, np.ndarray) or 
            (workspace.dtype != np.intc) or
            (workspace.shape[0] < t.shape[0]) or
            (workspace.shape[1] < s)):
        workspace = np.empty((t.shape[0], s), dtype=np.intc)
    
    ell = -1*np.ones(s, dtype=np.intc)

    # TODO: I am assuming memory is cheap and I don't get much for typing
    # the test array as bool_ vs intc
    back_slice = -max(k,1)
    test = np.empty((t.shape[0] - k + back_slice - 1, s), dtype=np.intc) # workspace[1:t.shape[0],:s]
    test[:,:s] = (t[k:back_slice-1,None] <= xvals[None,:]) & (xvals[None,:] < t[k+1:back_slice,None])
    if extrapolate:
        test[0,:s] = test[0,:s] | (xvals < t[k])
        test[-1,:s] = test[-1,:s] | (t[-k-1] <= xvals)

    # TODO: can we pre-allocate this? or is there a better way to implement
    # this whole function?
    where_test = np.nonzero(test)
    ell[where_test[1]] = where_test[0] + k
    return ell

def evaluate_spline(t, k, xvals, nu, extrapolate, 
    interval_workspace, 
    basis_workspace):
    """
    Evaluate the k+1 non-zero B-spline basis functions for xvals.

    Parameters
    ----------
    t : ndarray, shape (n+k+1)
        Knots of spline to evaluate.
    k : int
        Degree of spline to evaluate.
    xvals : ndarray, shape (s,)
        Points at which to evaluate the spline. 
    nu : int
        Order of derivative to evaluate.
    extrapolate : int, optional
        Whether to extrapolate to ouf-of-bounds points, or to return NaNs.
    interval_workspace : ndarray, shape (s,), dtype=int
        Array used to return identified intervals, modified in-place.
    basis_workspace : ndarray, shape (s, 2*k+2), dtype=float
        Array used to return computed values of the k+1 spline basis function
        at each of the input points, modified in-place.

    Notes
    -----
    This is a vectorized, NumPy implementation similar to the the Cython 
    `_bspl.evaluate_spline`.
    """

    if xvals.ndim != 1:
        raise ValueError("expected 1-dimensional xvals")

    s = xvals.size

    if (not isinstance(interval_workspace, np.ndarray) or 
            (interval_workspace.dtype != np.intc) or
            (interval_workspace.shape[0] < s)):
        raise ValueError("interval_workspace has invalid shape or dtype")
    ell = find_interval(t, k, xvals, extrapolate)
    
    
    basis_workspace = basis_workspace.T
    if (not isinstance(basis_workspace, np.ndarray) or 
            (basis_workspace.dtype != np.float_) or
            (basis_workspace.shape[0] < 2*k+2) or
            (basis_workspace.shape[1] < s)):
        raise ValueError("basis_workspace has invalid shape or dtype")

    u = basis_workspace[:k+1,:s]
    w = basis_workspace[k+1:2*k+2,:s]
    bounds = np.empty((2,s), dtype=np.float_)

    u[0,...] = 1.0
    for j in range(1, k-nu+1):
        w[:j] = u[:j].copy()
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
            xx = xvals[neq_test]
            tau = w[n-1, neq_test]/(xb-xa)
            u[n-1, neq_test] += tau*(xb - xx)
            u[n, neq_test] = tau*(xx - xa)

    for j in range(k-nu+1, k+1):
        w[:j] = u[:j].copy()
        u[0,:] = 0
        for n in range(1, j+1):
            index = ell+n
            bounds[0, :] = t[index]
            bounds[1, :] = t[index-j]
            neq_test = bounds[0, :] != bounds[1, :]
            u[nu, ~neq_test] = 0.0

            xb = bounds[0, neq_test]
            xa = bounds[1, neq_test]
            tau = j*w[n-1, neq_test]/(xb-xa)
            u[n-1, neq_test] -= tau
            u[n, neq_test] = tau

    interval_workspace[:s] = ell - k
