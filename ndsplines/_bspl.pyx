"""
Routines for evaluating and manipulating B-splines.

"""

from __future__ import absolute_import

import numpy as np
cimport numpy as cnp

cimport cython

cdef extern from "_bspl.h":
    void _deBoor_D(double *t, double x, int k, int ell, int m, double *result) nogil

cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"

ctypedef double complex double_complex

ctypedef fused double_or_complex:
    double
    double complex

# test
#------------------------------------------------------------------------------
# B-splines
#------------------------------------------------------------------------------

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int find_interval(const double[::1] t,
                       int k,
                       double xval,
                       int prev_l,
                       bint extrapolate) nogil:
    """
    Find an interval such that t[interval] <= xval < t[interval+1].

    Uses a linear search with locality, see fitpack's splev.

    Parameters
    ----------
    t : ndarray, shape (nt,)
        Knots
    k : int
        B-spline degree
    xval : double
        value to find the interval for
    prev_l : int
        interval where the previous value was located.
        if unknown, use any value < k to start the search.
    extrapolate : int
        whether to return the last or the first interval if xval
        is out of bounds.

    Returns
    -------
    interval : int
        Suitable interval or -1 if xval was nan.

    """
    cdef:
        int l
        int n = t.shape[0] - k - 1
        double tb = t[k]
        double te = t[n]

    if xval != xval:
        # nan
        return -1

    if ((xval < tb) or (xval > te)) and not extrapolate:
        return -1

    l = prev_l if k < prev_l < n else k

    # xval is in support, search for interval s.t. t[interval] <= xval < t[l+1]
    while(xval < t[l] and l != k):
        l -= 1

    l += 1
    while(xval >= t[l] and l != n):
        l += 1

    return l-1


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def evaluate_spline(const double[::1] t,
             int k,
             const double[::1] xp,
             int nu,
             bint extrapolate,
             int[::1] interval_workspace,
             double[:, ::1] basis_workspace):
    """
    Evaluate a spline in the B-spline basis.

    Parameters
    ----------
    t : ndarray, shape (n+k+1)
        knots
    k : int
        spline order
    xp : ndarray, shape (s,)
        Points to evaluate the spline at.
    nu : int
        Order of derivative to evaluate.
    extrapolate : int, optional
        Whether to extrapolate to ouf-of-bounds points, or to return NaNs.
    interval_workspace : ndarray, shape (s,), dtype=int
        Array of identified intervals
    basis_workspace : ndarray, shape (s, 2*k+3), dtype=float
        Computed values of the spline basis function at each of the input 
        points. This argument is modified in-place.

    """

    cdef int ip, jp
    cdef int interval
    cdef double xval

    # shape checks
    if interval_workspace.shape[0] < xp.shape[0]:
        raise ValueError("basis_workspace and xp have incompatible shapes")
    if basis_workspace.shape[0] < xp.shape[0]:
        raise ValueError("basis_workspace and xp have incompatible shapes")
    if basis_workspace.shape[1] != 2*k+3:
        raise ValueError("basis_workspace and k have incompatible shapes")

    # check derivative order
    if nu < 0:
        raise NotImplementedError("Cannot do derivative order %s." % nu)

    # evaluate
    with nogil:
        interval = k
        for ip in range(xp.shape[0]):
            xval = xp[ip]

            # Find correct interval
            interval = find_interval(t, k, xval, interval, extrapolate)

            if interval < 0:
                # xval was nan etc
                for jp in range(k+1):
                    basis_workspace[ip, jp] = nan
                continue

            # Evaluate (k+1) b-splines which are non-zero on the interval.
            # on return, first k+1 elemets of work are B_{m-k},..., B_{m}
            _deBoor_D(&t[0], xval, k, interval, nu, &basis_workspace[ip, 0])
            interval_workspace[ip] = interval - k

