import numpy as np

def find_intervals(t, k, x, extrapolate=False, workspace=None):
    """
    Find an interval ell such that t[ell] <= x < t[ell+1].

    Parameters
    ----------
    t : ndarray, shape=(n+k+1,) dtype=np.float_
        knots
    k : int
        order of B-spline
    x : ndarray, shape=(s,) dtype=np.float_
        values to find the interval for
    extrapolate : bool
        whether to return the last or the first interval if xval
        is out of bounds.

    Returns
    -------
    ell : ndarray, shape=(s,) dtype=np.intc
        Suitable interval or -1 for each value in x

    Notes
    -----
    Similar to scipy\\interpolate\\_bspl.pyx find_Interval (inline)
    """
    if x.ndim != 1:
        raise ValueError("expected 1-dimensional x")

    s = x.size

    do_return = False
    if (not isinstance(workspace, np.ndarray) or 
            (workspace.dtype != np.intc) or
            (workspace.shape[0] < t.shape[0]) or
            (workspace.shape[1] < s)):
        workspace = np.empty((t.shape[0], s), dtype=np.intc)
        do_return = True
    
    ell = workspace[0,:s]

    # TODO: I am assuming memory is cheap and I don't get much for typing
    # the test array as bool_ vs intc
    test = workspace[1:t.shape[0],:s]

    ell[:] = -1
    test[:,:s] = (t[:-1,None] <= x[None,:]) & (x[None,:] < t[1:,None])
    
    if extrapolate:
        test[k,:s] = test[k,:s] | (x < t[k])
        test[-k-1,:s] = test[-k-1,:s] | (t[-k-1] <= x)

    # TODO: can we pre-allocate this? or is there a better way to implement
    # this whole function?
    where_test = np.nonzero(test)
    ell[where_test[1]] = where_test[0]

    if do_return:
        return ell

def evaluate_spline(t, k, x, nu=0, extrapolate=False, 
    interval_workspace=None, 
    basis_workspace=None):
    """
    Evaluate the k+1 non-zero spline basis functions for x

    Parameters
    ----------
    t : ndarray, shape=(n+k+1,) dtype=np.float_
        knots
    k : int
        order of B-spline.
    x : ndarray, shape=(s,) dtype=np.float_
        values to find the interval for
    ell : ndarray, shape=(s,) dtype=np.intc
        index such that t[ell] <= x < t[ell+1] for each x
    nu : int
        order of derivative to evaluate.

    Returns
    -------
    u : ndarray, shape=(k+1, s) dtype=np.float_
        the values of the non-zero spline basis functions evaluated at x

    Notes
    -----
    similar to scipy\\interpolate\\src\\__fitpack.h _deBoor_D (inline)
    """

    if x.ndim != 1:
        raise ValueError("expected 1-dimensional x")

    s = x.size

    if (not isinstance(interval_workspace, np.ndarray) or 
            (interval_workspace.dtype != np.intc) or
            (interval_workspace.shape[0] < s)):
        raise ValueError("interval_workspace has invalid shape or dtype")
    ell = find_intervals(t, k, x, extrapolate)

    workspace = basis_workspace.T

    do_return = False
    if (not isinstance(workspace, np.ndarray) or 
            (workspace.dtype != np.float_) or
            (workspace.shape[0] < 2*k+3) or
            (workspace.shape[1] < s)):
        raise ValueError("basis_workspace has invalid shape or dtype")

    u = workspace[:k+1,:s]
    w = workspace[k+1:2*k+1,:s]
    bounds = workspace[2*k+1:2*k+3,:s]
    
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

    interval_workspace[:s] = ell - k

    if do_return:
        return u
