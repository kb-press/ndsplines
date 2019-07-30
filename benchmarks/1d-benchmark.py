"""
==================================
1D ndsplines vs. scipy.interpolate
==================================
"""

import numpy as np
import gc
import time

from scipy import interpolate
import ndsplines

import matplotlib.pyplot as plt

# number of time measurements per input/query size
n_iter = 10


def timeit(func, n_iter=1, return_samps=True, **func_kwargs):
    results = np.empty(n_iter, dtype=np.double)
    for i in range(n_iter):
        # gc.collect()

        tstart = time.time()
        func(**func_kwargs)
        delta = time.time() - tstart

        results[i] = delta

    if return_samps:
        return results
    else:
        return np.mean(results)


def gen_xy(size):
    x = np.pi * np.linspace(-1, 1, size)
    y = np.sin(x)
    return x, y


def gen_xx(size):
    return 3 * np.pi * np.linspace(-1, 1, size)


# make_interp_spline timing
x_sizes = np.logspace(1, 3, 10, dtype=int)
t_scipy_build = np.empty((2, x_sizes.size))
t_ndspl_build = np.empty((2, x_sizes.size))
for i, size in enumerate(x_sizes):
    x, y = gen_xy(size)
    t_scipy = 10e3 * timeit(interpolate.make_interp_spline, x=x.copy(), y=y,
                            n_iter=n_iter)
    t_ndspl = 10e3 * timeit(ndsplines.make_interp_spline, x=x.copy(), y=y,
                            n_iter=n_iter)
    t_scipy_build[:, i] = np.mean(t_scipy), np.std(t_scipy)
    t_ndspl_build[:, i] = np.mean(t_ndspl), np.std(t_ndspl)

# spline query timing
x, y = gen_xy(7)
xx_sizes = np.logspace(0, 3, 10, dtype=int)
t_scipy_call = np.empty((2, xx_sizes.size))
t_ndspl_npy_call = np.empty((2, xx_sizes.size))
t_ndspl_pyx_call = np.empty((2, xx_sizes.size))
for i, size in enumerate(xx_sizes):
    xx = gen_xx(size)
    spl_scipy = interpolate.make_interp_spline(x.copy(), y)
    spl_ndspl = ndsplines.make_interp_spline(x.copy(), y)
    spl_ndspl.allocate_workspace_arrays(size)
    t_scipy = 10e3 * timeit(spl_scipy, x=xx.copy(), n_iter=n_iter)
    ndsplines.set_impl('cython')
    t_ndspl_pyx = 10e3 * timeit(spl_ndspl, x=xx.copy(), n_iter=n_iter)
    ndsplines.set_impl('numpy')
    t_ndspl_npy = 10e3 * timeit(spl_ndspl, x=xx.copy(), n_iter=n_iter)
    t_scipy_call[:, i] = np.mean(t_scipy), np.std(t_scipy)
    t_ndspl_npy_call[:, i] = np.mean(t_ndspl_npy), np.std(t_ndspl_npy)
    t_ndspl_pyx_call[:, i] = np.mean(t_ndspl_pyx), np.std(t_ndspl_pyx)

# plot results
fig, axes = plt.subplots(nrows=2)

axes[0].errorbar(x_sizes, t_scipy_build[0], capsize=3, yerr=t_scipy_build[1],
                 label='scipy')
axes[0].errorbar(x_sizes, t_ndspl_build[0], capsize=3, yerr=t_ndspl_build[1],
                 label='ndsplines')
axes[0].set_title('make_interp_spline')

axes[1].errorbar(xx_sizes, t_scipy_call[0], capsize=3, yerr=t_scipy_call[1],
                 label='scipy.interpolate')
axes[1].errorbar(xx_sizes, t_ndspl_npy_call[0], capsize=3, yerr=t_ndspl_npy_call[1],
                 label='ndsplines npy')
axes[1].errorbar(xx_sizes, t_ndspl_pyx_call[0], capsize=3, yerr=t_ndspl_pyx_call[1],
                 label='ndsplines pyx')
axes[1].set_title('spline.__call__')

for ax in axes:
    ax.set_xlabel('input array size')
    ax.set_ylabel('time [ms]')
    ax.set_xscale('log')
    ax.grid()

axes[-1].legend()
fig.tight_layout()

plt.show()
