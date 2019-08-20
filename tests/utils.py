import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
np.random.seed(123)

def get_query_points(ndspline, n=1024):
    knot_minmax = np.empty((2, ndspline.xdim))
    for i in range(ndspline.xdim):
        k = ndspline.degrees[i]*0
        knot_minmax[0, i] = ndspline.knots[i][k]
        knot_minmax[1, i] = ndspline.knots[i][-k-1]
    query_points = np.random.rand(n, ndspline.xdim,)
    query_points[...] = query_points[...] * np.diff(knot_minmax, axis=0) + knot_minmax[0]
    return query_points


def assert_equal_splines(b_left, b_right):
    """ assert all properties of spline are equal (within tolerance) """
    assert b_left == b_right

def _make_random_spline(xdim=1, k=None, periodic=False, extrapolate=True, yshape=None, ydim=1, ymax=10):
    ns = []
    ts = []
    if k is None:
        ks = np.random.randint(5, size=xdim)
    else:
        ks = np.broadcast_to(k, (xdim,))
    if periodic is None:
        periodic = np.random.randint(2,size=xdim, dtype=np.bool_)
    if extrapolate is None:
        extrapolate = np.random.randint(2,size=xdim, dtype=np.bool_)

    if ydim is None:
        ydim = np.random.randint(5)
    if yshape is None:
        yshape = tuple(np.random.randint(1, ymax, size=ydim))
    for i in range(xdim):
        ns.append(np.random.randint(2*min(ks[i]+1,3),11))
        ts.append(np.sort(np.random.rand(ns[i]+ks[i]+1)))
    c = np.random.rand(*ns,*yshape)
    return ndsplines.NDSpline(ts, c, ks, periodic, extrapolate)

def get_grid_data(*nums):
    return np.stack(np.meshgrid(*[
        np.sort(np.random.rand(n)) for n in nums],
         indexing='ij'), axis=-1)
