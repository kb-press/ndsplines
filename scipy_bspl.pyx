"""
Routines for evaluating and manipulating B-splines.

"""

from __future__ import absolute_import

import numpy as np
cimport numpy as cnp

cimport cython

cdef extern from "scipy_bspl.h":
    void _deBoor_D(double *t, double x, int k, int ell, int m, double *result) nogil

cdef extern from "numpy/npy_math.h":
    double nan "NPY_NAN"

ctypedef double complex double_complex

ctypedef fused double_or_complex:
    double
    double complex


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
             double_or_complex[:, ::1] c,
             int k,
             const double[::1] xp,
             int nu,
             bint extrapolate,
             double_or_complex[:, ::1] out):
    """
    Evaluate a spline in the B-spline basis.

    Parameters
    ----------
    t : ndarray, shape (n+k+1)
        knots
    c : ndarray, shape (n, m)
        B-spline coefficients
    xp : ndarray, shape (s,)
        Points to evaluate the spline at.
    nu : int
        Order of derivative to evaluate.
    extrapolate : int, optional
        Whether to extrapolate to ouf-of-bounds points, or to return NaNs.
    out : ndarray, shape (s, m)
        Computed values of the spline at each of the input points.
        This argument is modified in-place.

    """

    cdef int ip, jp, n, a
    cdef int i, interval
    cdef double xval

    # shape checks
    if out.shape[0] != xp.shape[0]:
        raise ValueError("out and xp have incompatible shapes")
    if out.shape[1] != c.shape[1]:
        raise ValueError("out and c have incompatible shapes")

    # check derivative order
    if nu < 0:
        raise NotImplementedError("Cannot do derivative order %s." % nu)

    n = c.shape[0]
    cdef double[::1] work = np.empty(2*k+2, dtype=np.float_)

    # evaluate
    with nogil:
        interval = k
        for ip in range(xp.shape[0]):
            xval = xp[ip]

            # Find correct interval
            interval = find_interval(t, k, xval, interval, extrapolate)

            if interval < 0:
                # xval was nan etc
                for jp in range(c.shape[1]):
                    out[ip, jp] = nan
                continue

            # Evaluate (k+1) b-splines which are non-zero on the interval.
            # on return, first k+1 elemets of work are B_{m-k},..., B_{m}
            _deBoor_D(&t[0], xval, k, interval, nu, &work[0])

            # Form linear combinations
            for jp in range(c.shape[1]):
                out[ip, jp] = 0.
                for a in range(k+1):
                    out[ip, jp] = out[ip, jp] + c[interval + a - k, jp] * work[a]