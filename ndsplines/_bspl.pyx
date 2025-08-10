# cython: language_level=3
"""
Cython implementation for evaluating B-splines.

"""

# This file contains code from SciPy which has been adapted to suit the needs
# of ndsplines (scipy/interpolate/_bspl.pyx). See LICENSE for details.

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
    t : ndarray, shape (n+k+1)
        Knots of spline to evaluate.
    k : int
        Degree of spline to evaluate.
    xval : double
        Point at which to evaluate the spline. 
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
             const double[::1] xvals,
             int nu,
             bint extrapolate,
             int[::1] interval_workspace,
             double[:, ::1] basis_workspace):
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

    """

    cdef int ip, jp
    cdef int interval
    cdef double xval

    # shape checks
    if interval_workspace.shape[0] < xvals.shape[0]:
        raise ValueError("interval_workspace and xvals have incompatible shapes")
    if basis_workspace.shape[0] < xvals.shape[0]:
        raise ValueError("basis_workspace and xvals have incompatible shapes")
    if basis_workspace.shape[1] < 2*k+2:
        raise ValueError("basis_workspace and k have incompatible shapes")

    # check derivative order
    if nu < 0:
        raise NotImplementedError("Cannot do derivative order %s." % nu)

    # evaluate
    with nogil:
        interval = k
        for ip in range(xvals.shape[0]):
            xval = xvals[ip]

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


@cython.wraparound(False)
@cython.boundscheck(False)
def _colloc(const double[::1] x, const double[::1] t, int k, double[::1, :] ab,
            int offset=0):
    """Build the B-spline collocation matrix.

    The collocation matrix is defined as :math:`B_{j,l} = B_l(x_j)`,
    so that row ``j`` contains all the B-splines which are non-zero
    at ``x_j``.

    The matrix is constructed in the LAPACK banded storage.
    Basically, for an N-by-N matrix A with ku upper diagonals and
    kl lower diagonals, the shape of the array Ab is (2*kl + ku +1, N),
    where the last kl+ku+1 rows of Ab contain the diagonals of A, and
    the first kl rows of Ab are not referenced.
    For more info see, e.g. the docs for the ``*gbsv`` routine.

    This routine is not supposed to be called directly, and
    does no error checking.

    Parameters
    ----------
    x : ndarray, shape (n,)
        sorted 1D array of x values
    t : ndarray, shape (nt + k + 1,)
        sorted 1D array of knots
    k : int
        spline order
    ab : ndarray, shape (2*kl + ku + 1, nt), F-order
        This parameter is modified in-place.
        On exit: zeroed out.
        On exit: B-spline collocation matrix in the band storage with
        ``ku`` upper diagonals and ``kl`` lower diagonals.
        Here ``kl = ku = k``.
    offset : int, optional
        skip this many rows

    """
    cdef int left, j, a, kl, ku, clmn
    cdef double xval

    kl = ku = k
    cdef double[::1] wrk = np.empty(2*k + 2, dtype=np.float64)

    # collocation matrix
    with nogil:
        left = k
        for j in range(x.shape[0]):
            xval = x[j]
            # find interval
            left = find_interval(t, k, xval, left, extrapolate=False)

            # fill a row
            _deBoor_D(&t[0], xval, k, left, 0, &wrk[0])
            # for a full matrix it would be ``A[j + offset, left-k:left+1] = bb``
            # in the banded storage, need to spread the row over
            for a in range(k+1):
                clmn = left - k + a
                ab[kl + ku + j + offset - clmn, clmn] = wrk[a]


@cython.wraparound(False)
@cython.boundscheck(False)
def _handle_lhs_derivatives(const double[::1]t, int k, double xval,
                            double[::1, :] ab,
                            int kl, int ku,
                            const cnp.npy_long[::1] deriv_ords,
                            int offset=0):
    """ Fill in the entries of the collocation matrix corresponding to known
    derivatives at xval.

    The collocation matrix is in the banded storage, as prepared by _colloc.
    No error checking.

    Parameters
    ----------
    t : ndarray, shape (nt + k + 1,)
        knots
    k : integer
        B-spline order
    xval : float
        The value at which to evaluate the derivatives at.
    ab : ndarray, shape(2*kl + ku + 1, nt), Fortran order
        B-spline collocation matrix.
        This argument is modified *in-place*.
    kl : integer
        Number of lower diagonals of ab.
    ku : integer
        Number of upper diagonals of ab.
    deriv_ords : 1D ndarray
        Orders of derivatives known at xval
    offset : integer, optional
        Skip this many rows of the matrix ab.

    """
    cdef:
        int left, nu, a, clmn, row
        double[::1] wrk = np.empty(2*k+2, dtype=np.float64)

    # derivatives @ xval
    with nogil:
        left = find_interval(t, k, xval, k, extrapolate=False)
        for row in range(deriv_ords.shape[0]):
            nu = deriv_ords[row]
            _deBoor_D(&t[0], xval, k, left, nu, &wrk[0])
            # if A were a full matrix, it would be just
            # ``A[row + offset, left-k:left+1] = bb``.
            for a in range(k+1):
                clmn = left - k + a
                ab[kl + ku + offset + row - clmn, clmn] = wrk[a]

