import pytest
import ndsplines
import numpy as np


@pytest.mark.parametrize('impl', ('cython', 'numpy'))
def test_highlevel(impl):
    ndsplines.set_impl(impl)

    x = np.linspace(0, 1, 10)
    xx = np.linspace(0, 1, 100)
    y = np.sin(2*np.pi*5*x)
    spl = ndsplines.make_interp_spline(x, y)
    spl(xx)
