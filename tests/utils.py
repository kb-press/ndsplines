import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
np.random.seed(123)

def get_query_points(ndspline, n=1024):
    knot_minmax = np.empty((2, ndspline.xdim))
    knot_minmax[0, :] = 0.0
    knot_minmax[1, :] = 1.0
    # for i in range(ndspline.xdim):
    #     k = ndspline.degrees[i]*1
    #     knot_minmax[0, i] = 0.0 #ndspline.knots[i][k]
    #     knot_minmax[1, i] = 1.0 #ndspline.knots[i][-k-1]
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
        periodic = np.random.randint(2,size=xdim, dtype=bool)
    if extrapolate is None:
        extrapolate = np.random.randint(2,size=xdim, dtype=bool)

    if ydim is None:
        ydim = np.random.randint(5)
    if yshape is None:
        yshape = tuple(np.random.randint(1, ymax, size=ydim))
    for i in range(xdim):
        ns.append(np.random.randint(7) + 2*ks[i] + 3)
        ts.append(np.r_[0.0:0.0:(ks[i]+1)*1j,
            np.sort(np.random.rand(ns[i]-ks[i]-1)),
            1.0:1.0:(ks[i]+1)*1j
            ])
    c = np.random.rand(*ns,*yshape)
    return ndsplines.NDSpline(ts, c, ks, periodic, extrapolate)

def un_knot_a_knot(knots, degrees):
    return [np.r_[
            t[0], 
            t[0] + (t[k+1]-t[0])/max(k,3):t[k+1] - (t[k+1]-t[0])/max(k,3):max(k-2,0)*1j,
            t[k+1:-k-1],
            t[-k-2] + (t[-1]-t[-k-2])/max(k,3):t[-1] - (t[-1]-t[-k-2])/max(k,3):max(k-2,0)*1j,
            t[-1]
        ].squeeze() 
         for t, k in zip(knots, degrees) if k == 3]

def get_grid_data(*nums):
    return np.stack(np.meshgrid(*[
        np.sort(np.random.rand(n)) for n in nums],
         indexing='ij'), axis=-1)
