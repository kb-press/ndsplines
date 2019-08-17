import pytest
import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import interpolate
from utils import (get_query_points, assert_equal_splines, _make_random_spline,
    copy_ndspline)

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1),
    _make_random_spline(1),
    _make_random_spline(1),
    _make_random_spline(1),
])
def test_1d_make_lsq(ndspline):
    N = 100
    sample_x = np.sort(get_query_points(ndspline, n=N).squeeze())
    sample_y = ndspline(sample_x) 
    # it was non-trivial to figure out the proper parameters for
    # scipy.interpolate. It needed specific knot sequence (possibly other 
    # solutions) and sorted sample data. ndspline did not need either.
    for k in range(4):
        knots = np.r_[(0.0,)*(k+1), 0.25, 0.5, 0.75, (1.0,)*(k+1)]

        # unweighted
        nspl = ndsplines.make_lsq_spline(sample_x, sample_y, [knots], [k])
        ispl = interpolate.make_lsq_spline(sample_x, sample_y, knots, k)
        assert_allclose(nspl.coefficients.reshape(-1), ispl.c.reshape(-1))

        # random weights
        w = np.random.random(N)
        nspl = ndsplines.make_lsq_spline(sample_x, sample_y, [knots], [k], w)
        ispl = interpolate.make_lsq_spline(sample_x, sample_y, knots, k, w)
        assert_allclose(nspl.coefficients.reshape(-1), ispl.c.reshape(-1))

# 
# construct a valid spline. We expect this to fail.
@pytest.mark.skip(reason="``interpolate.LSQBivariateSpline`` seems buggy: and does not always construct valid splines and sometimes segfaults.")
@pytest.mark.parametrize('ndspline', [
    # I believe LSQBivariateSpline requires 1-D output
    _make_random_spline(2, yshape=(1,)),
    _make_random_spline(2, yshape=(1,)),
    _make_random_spline(2, yshape=(1,)),
    _make_random_spline(2, yshape=(1,)),
])
def test_2d_make_lsq(ndspline):
    N = 500
    sample_x = get_query_points(ndspline, n=N).squeeze()
    sample_y = ndspline(sample_x)
    for kx in range(2,4):
        knots_x = np.r_[(0.0,)*(kx+1), 0.25, 0.5, 0.75, (1.0,)*(kx+1)]
        for ky in range(2,4):
            knots_y = np.r_[(0.0,)*(ky+1), 0.25, 0.5, 0.75, (1.0,)*(ky+1)]

            knots = [knots_x, knots_y]

            # unweighted
            nspl = ndsplines.make_lsq_spline(sample_x, sample_y, knots, [kx, ky])
            ispl = interpolate.LSQBivariateSpline(
                sample_x[:, 0],
                sample_x[:, 1], 
                sample_y, 
                knots_x[kx+1:-kx-1], knots_y[ky+1:-ky-1], 
                bbox=[0.,1.,0.,1.],
                kx=kx, ky=ky)
            assert_allclose(ispl.get_knots()[0], nspl.knots[0])
            assert_allclose(ispl.get_knots()[1], nspl.knots[1])
            assert_allclose(nspl.coefficients.reshape(-1), ispl.get_coeffs().reshape(-1))

            # random weights
            w = np.random.random(N)
            nspl = ndsplines.make_lsq_spline(sample_x, sample_y, knots, [kx, ky], w)
            ispl = interpolate.LSQBivariateSpline(
                sample_x[:, 0],
                sample_x[:, 1], 
                sample_y, 
                knots_x[kx+1:-kx-1], knots_y[ky+1:-ky-1], 
                w=w,
                bbox=[0.,1.,0.,1.],
                kx=kx, ky=ky, )
            assert_allclose(ispl.get_knots()[0], nspl.knots[0])
            assert_allclose(ispl.get_knots()[1], nspl.knots[1])
            assert_allclose(nspl.coefficients.reshape(-1), ispl.get_coeffs().reshape(-1))

