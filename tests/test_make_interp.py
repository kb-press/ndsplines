import pytest
import ndsplines
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_equal
from scipy import interpolate
from utils import (get_query_points, assert_equal_splines, _make_random_spline,
    get_grid_data)

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1),
    _make_random_spline(1),
    _make_random_spline(1),
    _make_random_spline(1),
])
def test_1d_make_interp(ndspline):
    for k in range(4):
        N=np.random.randint(k+1, 35)
        sample_x = get_grid_data(N).reshape(-1)
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
        nx = np.random.randint(4, 35)

        for ky in range(1,4):
            ny = np.random.randint(4, 35)

            sample_x = get_grid_data(nx, ny).squeeze()
            sample_y = ndspline(sample_x).squeeze()
            nspl = ndsplines.make_interp_spline(sample_x, sample_y, [kx, ky])
            ispl = interpolate.RectBivariateSpline(sample_x[:, 0, 0], sample_x[0, :, 1], sample_y, kx=kx, ky=ky)
            assert_allclose(nspl.coefficients.reshape(-1), ispl.get_coeffs().reshape(-1))


@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1, periodic=None, extrapolate=None),
    _make_random_spline(2, periodic=None, extrapolate=None),
    _make_random_spline(3, periodic=None, extrapolate=None),
    _make_random_spline(4, periodic=None, extrapolate=None),
])
def test_nd_make_interp(ndspline):
    sample_x = get_grid_data(*[knot.size for knot in ndspline.knots])
    sample_y = ndspline(sample_x)

    k = 3
    # TODO: figure out how to loop

    # does it interpolate?
    nspl = ndsplines.make_interp_spline(sample_x, sample_y, k)
    assert_allclose(sample_y, nspl(sample_x))
    
    # can you re-create this spline?
    knot_sample_x = np.stack(np.meshgrid(
        *[np.r_[t[0], (t[0]+t[k+1])/2, t[k+1:-k-1], (t[-k-2]+t[-1])/2, t[-1]] 
         for t in nspl.knots],
        indexing='ij'), axis=-1)
    knot_sample_y = nspl(knot_sample_x)

    nspl2 = ndsplines.make_interp_spline(sample_x, sample_y, k)
    assert_equal_splines(nspl, nspl2)

    # can you use a tidy dataformat?
    tidy_x = knot_sample_x.reshape((-1, nspl.xdim))
    tidy_y = knot_sample_y.reshape((-1, nspl.ydim))
    tidy_array = np.concatenate((tidy_x, tidy_y), axis=1)
    x_sel = np.arange(nspl.xdim)
    y_sel = np.arange(nspl.ydim) + nspl.xdim
    tidy_df = pd.DataFrame(tidy_array, columns=['column %d' % i for i in range(tidy_array.shape[1])])
    nspl3 = ndsplines.make_interp_spline_from_tidy(tidy_array, x_sel, y_sel)
    assert_equal_splines(nspl, nspl3)

    tidy_df_x_col = tidy_df.columns[x_sel]
    tidy_df_y_col = tidy_df.columns[y_sel]
    nspl4 = ndsplines.make_interp_spline_from_tidy(tidy_df, tidy_df_x_col, tidy_df_y_col)
    assert_equal_splines(nspl, nspl4)

