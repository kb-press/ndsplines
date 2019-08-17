import pytest
import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import interpolate
from utils import (get_query_points, assert_equal_splines, _make_random_spline,
    copy_ndspline)

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1, 0, ydim=1),
    # _make_random_spline(1, 1, ydim=1),
    # _make_random_spline(1, 2, ydim=1),
    _make_random_spline(1, 3, ydim=1),
])
def test_1d_eval(ndspline):
    """ compare evaluation of 1-d NDSpline to scipy.interpolate.BSpline """
    bn = ndspline
    query_points = get_query_points(bn)
    bs = interpolate.BSpline(bn.knots[0], bn.coefficients, bn.degrees[0])
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
    _make_random_spline(2, [kx, ky], yshape=()) for kx in range(1,4) for ky in range(1,4)
])
def test_2d_eval(ndspline):
    """ compare evaluation of 2-d NDSpline to scipy.interpolate.BivariateSpline """
    bn = ndspline
    query_points = get_query_points(bn)
    bs = interpolate.BivariateSpline._from_tck(
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
            bs_res = bs(query_points[:,0], query_points[:,1], nux, 0, grid=False)

            assert_allclose(bn_res, bs_res)

        for nuy in range(1, bn.degrees[1]):

            bn_res = bn(query_points, np.r_[0, nuy]).squeeze()
            bs_res = bs(query_points[:,0], query_points[:,1], 0, nuy, grid=False)

            assert_allclose(bn_res, bs_res)
