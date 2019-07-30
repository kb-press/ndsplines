"""
==================================
2D ndsplines vs. scipy.interpolate
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


def gen_xyz(sizex, sizey):
    x = np.pi * np.linspace(-1, 1, sizex)
    y = np.pi * np.linspace(-1, 1, sizey)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    zz = np.sin(xx) * np.cos(yy)
    return x, y, zz


def gen_xxyy(sizex, sizey):
    x = np.pi * np.linspace(-1, 1, sizex)
    y = np.pi * np.linspace(-1, 1, sizey)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    return xx, yy


# make_interp_spline timing
x_sizes = np.logspace(1, 3, 10, dtype=int)
t_scipy_build = np.empty((2, x_sizes.size))
t_ndspl_build = np.empty((2, x_sizes.size))
for i, size in enumerate(x_sizes):
    x, y, z = gen_xyz(size, size)
    t_scipy = 10e3 * timeit(interpolate.RectBivariateSpline, x=x.copy(), y=y.copy(), z=z.copy(), 
                            n_iter=n_iter)
    t_ndspl = 10e3 * timeit(ndsplines.make_interp_spline, x=[x,y], y=z,
                            n_iter=n_iter)
    t_scipy_build[:, i] = np.mean(t_scipy), np.std(t_scipy)
    t_ndspl_build[:, i] = np.mean(t_ndspl), np.std(t_ndspl)

# spline query timing
x, y, z = gen_xyz(7, 5)
xx_sizes = np.logspace(0, 2, 10, dtype=int)
t_scipy_call = np.empty((2, xx_sizes.size))
t_ndspl_npy_call = np.empty((2, xx_sizes.size))
t_ndspl_pyx_call = np.empty((2, xx_sizes.size))
for i, size in enumerate(xx_sizes):
    xx, yy = gen_xxyy(size, size)
    xxyy = np.stack((xx, yy), axis=-1)
    spl_scipy = interpolate.RectBivariateSpline(x.copy(), y.copy(), z)
    spl_ndspl = ndsplines.make_interp_spline((x,y), z)
    spl_ndspl.allocate_workspace_arrays(size)
    t_scipy = 10e3 * timeit(spl_scipy, x=xx.copy(), y=yy.copy(), grid=False, n_iter=n_iter)
    ndsplines.set_impl('cython')
    t_ndspl_pyx = 10e3 * timeit(spl_ndspl, x=xxyy,
                            n_iter=n_iter)
    ndsplines.set_impl('numpy')
    t_ndspl_npy = 10e3 * timeit(spl_ndspl, x=xxyy,
                            n_iter=n_iter)
    t_scipy_call[:, i] = np.mean(t_scipy), np.std(t_scipy)
    t_ndspl_pyx_call[:, i] = np.mean(t_ndspl_pyx), np.std(t_ndspl_pyx)
    t_ndspl_npy_call[:, i] = np.mean(t_ndspl_npy), np.std(t_ndspl_npy)

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
