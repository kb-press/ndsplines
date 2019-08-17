import pytest
import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import interpolate
from utils import (get_query_points, assert_equal_splines, _make_random_spline,
    copy_ndspline)

def get_grid_data(*nums):
    return np.stack(np.meshgrid(*[
        np.sort(np.random.rand(n,)) for n in nums],
         indexing='ij'), axis=-1)

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1),
    _make_random_spline(1),
    _make_random_spline(1),
    _make_random_spline(1),
])
def test_1d_make_interp(ndspline):
    for k in range(4):
        N=np.random.randint(k+1, 35)
        sample_x = get_grid_data(N).squeeze()
        sample_y = ndspline(sample_x)
        nspl = ndsplines.make_interp_spline(sample_x, sample_y, k)
        ispl = interpolate.make_interp_spline(sample_x, sample_y, k)
        assert_allclose(nspl.coefficients.reshape(-1), ispl.c.reshape(-1))

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(2, yshape=(1,)),
    _make_random_spline(2, yshape=(1,)),
    _make_random_spline(2, yshape=(1,)),
    _make_random_spline(2, yshape=(1,)),
])
def test_2d_make_interp(ndspline):
    for kx in range(1,4):
        nx = np.random.randint(kx+1, 35)

        for ky in range(1,4):
            ny = np.random.randint(kx+1, 35)

            sample_x = get_grid_data(nx, ny).squeeze()
            sample_y = ndspline(sample_x)
            nspl = ndsplines.make_interp_spline(sample_x, sample_y, [kx, ky])
            ispl = interpolate.RectBivariateSpline(sample_x[:, 0, 0], sample_x[0, :, 1], sample_y, kx=kx, ky=ky)
            assert_allclose(nspl.coefficients.reshape(-1), ispl.get_coeffs().reshape(-1))
