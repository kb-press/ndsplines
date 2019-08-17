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
def test_1d_make_interp(ndspline):
    N=10
    sample_x = np.sort(get_query_points(ndspline, n=N).squeeze())
    sample_y = ndspline(sample_x)
    for k in range(4):
        nspl = ndsplines.make_interp_spline(sample_x, sample_y, k)
        ispl = interpolate.make_interp_spline(sample_x, sample_y, k)
        assert_allclose(nspl.coefficients.reshape(-1), ispl.c.reshape(-1))

