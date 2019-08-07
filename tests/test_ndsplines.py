import pytest
import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.interpolate import BSpline, BivariateSpline
from scipy.stats import norm
import itertools

np.random.seed(123)

#
# Integration/Miscellaneous tests
#

def test_evaluate_spline_different_impls():
    """Check that setting the backend implementation is effective."""
    ndsplines.set_impl('numpy')
    f_numpy = ndsplines.evaluate_spline

    ndsplines.set_impl('cython')
    f_cython = ndsplines.evaluate_spline

    assert f_numpy is not f_cython


#
# Scipy compatibility tests
#

def test_make_interp_scipy_compat():
    """Basic test of compatibility with scipy.interpolate API."""
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    spl = ndsplines.make_interp_spline(x, y)
    spl(np.linspace(0, 1, 100))


# Input/output API tests

def test_make_interp_invalid_x():
    """Bad input raises ValueError."""
    with pytest.raises(ValueError):
        ndsplines.make_interp_spline('str', np.arange(3))


def test_make_interp_invalid_y():
    """Bad input raises ValueError."""
    with pytest.raises(ValueError):
        ndsplines.make_interp_spline(np.arange(10), np.zeros((9, 10, 10, 10)))


@pytest.mark.parametrize('n_vals', [[8, 16], [8, 10, 12]])
def test_make_interp_x_vectors(n_vals):
    """Check that a list of vectors is accepted for x.

    y input in this case should have shape (n_ndim1, n_ndim2, ...) as if it
    were sampled on the grid.
    """
    x = [np.linspace(0, 1, n) for n in n_vals]
    xgrid = np.stack(np.meshgrid(*x, indexing='ij'), axis=-1)
    y = np.random.rand(*n_vals)

    spl = ndsplines.make_interp_spline(x, y)

    assert spl.xdim == len(n_vals)
    assert spl.ydim == 1
    assert_allclose(spl(xgrid), y)


@pytest.mark.parametrize('n_vals', [[10], [10, 12], [10, 12, 15]])
def test_make_interp_x_mesh(n_vals):
    """Input x arrays of varying dimensionality."""
    xarrays = [np.linspace(0, 1, n) for n in n_vals]
    x = np.stack(np.meshgrid(*xarrays, indexing='ij'), axis=-1)
    y = np.random.rand(*n_vals)

    spl = ndsplines.make_interp_spline(x, y)
    assert spl.xdim == len(n_vals)

    xsamp = np.random.randn(10, len(n_vals))
    assert spl(xsamp).shape == (10,)


@pytest.mark.parametrize('ydim', [2, 3])
def test_make_interp_nd_y(ydim):
    """Multi-dimensional y."""
    x = np.linspace(0, 1, 10)
    y = np.random.rand(10, ydim)

    spl = ndsplines.make_interp_spline(x, y)

    assert spl.xdim == 1
    assert spl.ydim == ydim

    samps = spl(np.random.rand(20))
    assert samps.shape == (20, ydim)


def test_make_interp_1d_y():
    """Check that output is squeezed ndim==1 for 1D y."""
    x = np.linspace(0, 1, 10)
    y = np.sin(x)
    spl = ndsplines.make_interp_spline(x, y)
    assert spl(np.random.rand(20)).shape == (20,)


#
# Mathematical tests
#

def test_make_interp_nn():
    """Verify nearest neighbor special case."""
    dx = 0.1
    x = np.arange(0, 1, dx)
    y = np.sin(2*np.pi*x)

    spl = ndsplines.make_interp_spline(x, y, bcs=[(0, 0), (0, 0)], degrees=0)

    # samples at offsets less than dx/2 will be same as original values
    xx = x[:-1] + dx/4
    assert_allclose(spl(xx), spl(x[:-1]))

def get_query_points(ndspline):
    knot_minmax = np.empty((2, ndspline.xdim))
    for i in range(ndspline.xdim):
        knot_minmax[0, i] = ndspline.knots[i][0]
        knot_minmax[1, i] = ndspline.knots[i][-1]
    query_points = np.random.rand(1024, ndspline.xdim,)
    query_points[...] = query_points[...] * np.diff(knot_minmax, axis=0) + knot_minmax[0]
    return query_points


def assert_equal_splines(b_left, b_right):
    """ assert all properties of spline are equal (within tolerance) """
    for knot_left, knot_right in zip(b_left.knots, b_right.knots):
        assert_allclose(knot_left, knot_right)

    assert_allclose(b_left.coefficients, b_right.coefficients)

    # TODO: is this the best way to do this test?
    assert_equal(b_left.degrees, b_right.degrees)
    assert_equal(b_left.periodic, b_right.periodic)
    assert_equal(b_left.extrapolate, b_right.extrapolate)

def _make_random_spline(xdim=1, k=None, periodic=None, extrapolate=None):
    ns = []
    ts = []
    ks = np.broadcast_to(k, (xdim,))
    for i in range(xdim):
        ns.append(np.random.randint(2*ks[i]+1,35))
        ts.append(np.sort(np.random.rand(ns[i]+ks[i]+1))) # works for 1d, fails for 2d
        # ts.append(np.r_[0:0:(ks[i])*1j, 0:1:(ns[i]-ks[i]+1)*1j, 1:1:(ks[i])*1j]) # works for 2d
        # ts.append(np.r_[0:0:(ks[i]+1)*1j, np.sort(np.random.rand(ns[i]-ks[i]-1)), 1:1:(ks[i]+1)*1j])
    c = np.random.rand(*ns,1)
    return ndsplines.NDSpline(ts, c, ks)

def copy_ndspline(ndspline):
    return ndsplines.NDSpline(
        ndspline.knots,
        ndspline.coefficients,
        ndspline.degrees,
        ndspline.periodic,
        ndspline.extrapolate,
        )

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1, 0),
    _make_random_spline(1, 1),
    _make_random_spline(1, 2),
    _make_random_spline(1, 3),
    _make_random_spline(2, 0),
    _make_random_spline(2, 1),
    _make_random_spline(2, 2),
])
def test_calculus(ndspline):
    """ verify calculus properties """
    b = ndspline
    query_points = get_query_points(b)
    nus = np.zeros((b.xdim), dtype=np.int)
    for i in range(b.xdim):
        nus[i] = 1
        if b.degrees[i] < 1:
            with pytest.raises(ValueError):
                der_b_i = b.derivative(i, 1)
            continue

        der_b_i = b.derivative(i, 1)
        antider_b_i = b.antiderivative(i, 1)

        assert_equal_splines(antider_b_i, b.derivative(i, -1))
        assert_equal_splines(der_b_i, b.antiderivative(i, -1))

        assert_allclose(der_b_i(query_points), b(query_points, nus) ) 
        assert_allclose(b(query_points), antider_b_i(query_points, nus) )

        offset = np.random.rand()

        der_offset_antider_b_i = copy_ndspline(antider_b_i)
        der_offset_antider_b_i.coefficients = der_offset_antider_b_i.coefficients + offset

        antider_der_b_i = der_b_i.antiderivative(i, 1)
        der_antider_b_i = antider_b_i.derivative(i, 1)
        der_offset_antider_b_i = der_offset_antider_b_i.derivative(i, 1)

        assert_equal_splines(der_antider_b_i, b)
        assert_equal_splines(der_offset_antider_b_i, b)

        for j in range(b.xdim):
            if i == j:
                continue
            der_b_ij = der_b_i.derivative(j, 1)
            der_b_ji = b.derivative(j, 1).derivative(i, 1)
            assert_equal_splines(der_b_ij, der_b_ji)

        nus[i] = 0

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1, 0),
    _make_random_spline(1, 1),
    _make_random_spline(1, 2),
    _make_random_spline(1, 3),
    _make_random_spline(2, 0),
    _make_random_spline(2, 1),
    _make_random_spline(2, 2),
    _make_random_spline(2, 3),
])
def test_file_io(ndspline):
    """ verify lossless file i/o """
    b = ndspline
    fname = 'test_file_io.npz'
    b.to_file(fname, True)
    assert_equal_splines(b, ndsplines.from_file(fname))
    b.to_file(fname, False)
    assert_equal_splines(b, ndsplines.from_file(fname))

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1, 0),
    _make_random_spline(1, 1),
    _make_random_spline(1, 2),
    _make_random_spline(1, 3),
])
def test_1d_eval(ndspline):
    bn = ndspline
    query_points = get_query_points(bn)
    bs = BSpline(bn.knots[0], bn.coefficients, bn.degrees[0])
    for nu in range(bn.degrees[0]+1):
        bs.extrapolate = True
        bn.extrapolate = True

        bs_res = bs(query_points, nu).squeeze()
        bn_res = bn(query_points, nu).squeeze()
        assert_allclose(bn_res, bs_res)

        bs.extrapolate = False
        bn.extrapolate = False
        bs_res = bs(query_points, nu).squeeze()
        bn_res = bn(query_points, nu).squeeze()
        assert_allclose(bn_res[~np.isnan(bs_res)], bs_res[~np.isnan(bs_res)])

        bs.extrapolate = 'periodic'
        bn.periodic = True

        bs_res = bs(query_points, nu).squeeze()
        bn_res = bn(query_points, nu).squeeze()
        assert_allclose(bn_res, bs_res)


        
@pytest.mark.parametrize('ndspline', [
    _make_random_spline(2, [kx, ky]) for kx in range(4) for ky in range(4)
])
def test_2d_eval(ndspline):
    bn = ndspline
    query_points = get_query_points(bn)
    bs = BivariateSpline._from_tck(
        tuple(ndspline.knots + 
            [ndspline.coefficients.reshape(-1)] + 
            ndspline.degrees.tolist()
        ))

    bn.extrapolate = False

    bn_res = bn(query_points,).squeeze()
    bs_res = bs(query_points[:,0], query_points[:,1], grid=False)
    assert_allclose(bn_res, bs_res)

    if np.all(bn.degrees>0):


        for nux in range(1, bn.degrees[0]):
            bn_res = bn(query_points, np.r_[nux, 0]).squeeze()
            try:
                bs_res = bs(query_points[:,0], query_points[:,1], nux, 0, grid=False)
            except ValueError:
                print("nux:", nux, "xshape: ", ndspline.xshape, "degrees:", bn.degrees)
                # continue

            assert_allclose(bn_res, bs_res)

        for nuy in range(1, bn.degrees[1]):

            bn_res = bn(query_points, np.r_[0, nuy]).squeeze()
            try:
                bs_res = bs(query_points[:,0], query_points[:,1], 0, nuy, grid=False)
            except ValueError:
                print("nuy:", nuy, "xshape: ", ndspline.xshape, "degrees:", bn.degrees)
                # continue

            assert_allclose(bn_res, bs_res)


