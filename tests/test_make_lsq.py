import pytest
import ndsplines
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy import interpolate
from utils import (get_query_points, assert_equal_splines, _make_random_spline,
    get_grid_data, un_knot_a_knot)

@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1, k=0, yshape=(1,)),
    _make_random_spline(1, k=1, yshape=(1,)),
    _make_random_spline(1, k=2, yshape=(1,)),
])
def test_1d_make_lsq(ndspline):
    N = 100
    stddev = 1E-3
    sample_x = np.sort(get_query_points(ndspline, n=N).squeeze())
    sample_y = ndspline(sample_x)
    signal_rms = (sample_y**2).sum(axis=0)/N
    snr_ratio = 10
    sample_y = sample_y + (signal_rms/snr_ratio)*np.random.random(sample_y.shape)
    # it was non-trivial to figure out the proper parameters for
    # scipy.interpolate. It needed specific knot sequence (possibly other 
    # solutions) and sorted sample data. ndspline did not need either.
    for k in range(0, 4):
        knots = np.r_[(0.0,)*(k+1), 0.25, 0.5, 0.75, (1.0,)*(k+1)]

        # unweighted
        nspl = ndsplines.make_lsq_spline(sample_x, sample_y.copy(), [knots], [k])
        try:
            ispl = interpolate.make_lsq_spline(sample_x, sample_y.copy(), knots, k)
        except np.linalg.linalg.LinAlgError as e:
            if "leading minor" in e.__repr__():
                print(e)
        else:
            assert_allclose(nspl.coefficients.reshape(-1), ispl.c.reshape(-1))

        # random weights
        w = np.random.random(N)
        nspl = ndsplines.make_lsq_spline(sample_x, sample_y, [knots], [k], w)
        try:
            ispl = interpolate.make_lsq_spline(sample_x, sample_y, knots, k, w)
        except np.linalg.linalg.LinAlgError as e:
            if "leading minor" in e.__repr__():
                print(e)
        else:
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


@pytest.mark.parametrize('ndspline', [
    _make_random_spline(1, periodic=None, extrapolate=None),
    _make_random_spline(2, periodic=None, extrapolate=None),
    _make_random_spline(3, periodic=None, extrapolate=None),
])
def test_nd_make_lsq(ndspline):
    sample_x = get_grid_data(*[t.size-k-1 
            for t, k in zip(ndspline.knots, ndspline.degrees)])
    sample_y = ndspline(sample_x)

    k = 3
    nspl = ndsplines.make_interp_spline(sample_x, sample_y, k)

    knots_to_reproduce =un_knot_a_knot(nspl.knots, nspl.degrees)
    knot_sample_x = np.stack(np.meshgrid(
        *knots_to_reproduce,
        indexing='ij'), axis=-1)
    knot_sample_y = ndspline(knot_sample_x)

    nspl = ndsplines.make_interp_spline(knot_sample_x, knot_sample_y, k)

    sample_x = np.stack(np.meshgrid(*[
        np.linspace(0,1, int(1.5*nspl.xshape[i])) for i in range(nspl.xdim)],
         indexing='ij'), axis=-1).reshape((-1, nspl.xdim))

    sample_y = nspl(sample_x)

    nlsq = ndsplines.make_lsq_spline(sample_x, sample_y, nspl.knots, nspl.degrees)
    assert_allclose(nlsq.coefficients, nspl.coefficients, rtol=1E-4)
    
    sample_y_orig = sample_y
    signal_rms = (sample_y**2).sum(axis=0)/sample_x.size

    N_samples = 4
    set_snrs = np.empty(N_samples)
    eval_snrs = np.empty(N_samples)

    for snr_exp in range(N_samples):
        snr_ratio = 10**(0*nspl.xdim+snr_exp)
        sample_y = sample_y_orig + signal_rms[None, :]/snr_ratio*np.random.random(sample_y.shape)
        nlsq = ndsplines.make_lsq_spline(sample_x, sample_y, nspl.knots, nspl.degrees)
        eval_snrs[snr_exp] = np.max(np.abs(nlsq.coefficients - nspl.coefficients)/nspl.coefficients)
        set_snrs[snr_exp] = snr_ratio
    assert (np.diff(np.log10(eval_snrs)).mean() < np.diff(np.log10(set_snrs)).mean()/2)
    