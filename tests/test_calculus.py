import pytest
import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import interpolate
from utils import get_query_points, assert_equal_splines, _make_random_spline

@pytest.fixture(
    params=[_make_random_spline(1, kx,) for kx in range(4) ] + \
    [_make_random_spline(2, [kx, ky]) for kx in range(1,4) for ky in range(1,4)] + \
    [_make_random_spline(3, [kx, ky, kz]) 
       for kx in range(1,4) for ky in range(1,4) for kz in range(1,4)],
)
def next_ndspline(request):
    return request.param

@pytest.fixture(params=['cython', 'numpy'])
def impl(request):
    return request.param

def test_calculus(next_ndspline, impl):
    """ verify calculus properties """
    ndsplines.set_impl(impl)
    b = next_ndspline
    query_points = get_query_points(b)
    nus = np.zeros((b.xdim), dtype=int)
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

        der_offset_antider_b_i = antider_b_i.copy()
        der_offset_antider_b_i.coefficients = der_offset_antider_b_i.coefficients + offset

        antider_der_b_i = der_b_i.antiderivative(i, 1)
        der_antider_b_i = antider_b_i.derivative(i, 1)
        der_offset_antider_b_i = der_offset_antider_b_i.derivative(i, 1)

        assert_equal_splines(der_antider_b_i, b)
        assert_equal_splines(der_offset_antider_b_i, b)

        for j in range(b.xdim):
            if i == j or b.degrees[j] < 1:
                continue
            der_b_ij = der_b_i.derivative(j, 1)
            der_b_ji = b.derivative(j, 1).derivative(i, 1)
            assert_equal_splines(der_b_ij, der_b_ji)

        nus[i] = 0
    ndsplines.set_impl('cython')
