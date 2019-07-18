"""
===========================
2-Dimensional Interpolation
===========================
"""
import ndsplines
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm
import itertools
from mpl_toolkits.mplot3d import Axes3D


def gaussian(x_in):
    z = norm.ppf(.995)
    x = z*(2*x_in-1)
    return norm.pdf(x)

def sin(x_in):
    x = np.pi*(x_in-0.5)
    return np.sin(x)

def tanh(x_in):
    x = 2*np.pi*(x_in-0.5)
    return np.tanh(x)

def dist(x_in, y_in):
    return np.sqrt((x_in-0.25)**2 + (y_in-0.25)**2)

funcs = [gaussian, sin, tanh]

def wrap2d(funcx, funcy):
    def func2d(x_in, y_in):
        return funcx(x_in)*funcy(y_in)
    func2d.__name__ = '_'.join([funcx.__name__, funcy.__name__])
    return func2d

funcs = [ wrap2d(*funcs_to_wrap) for funcs_to_wrap in itertools.combinations_with_replacement(funcs, r=2)]
funcs.append(dist)

x = np.linspace(0, 1, 7)
y = np.linspace(0, 1, 7)

xx = np.linspace(0,1,64) 
yy = np.linspace(0,1,64)

xx = np.linspace(-.25, 1.25, 64)
yy = np.linspace(-.25, 1.25, 64)
k = 3


meshx, meshy = np.meshgrid(x, y, indexing='ij')
gridxy = np.stack((meshx, meshy), axis=-1)


meshxx, meshyy = np.meshgrid(xx, yy, indexing='ij')
gridxxyy = np.stack((meshxx, meshyy), axis=-1)

for func in funcs:
    fvals = func(meshx, meshy)
    truef = func(meshxx, meshyy)
    test_NDBspline = ndsplines.make_interp_spline(gridxy, fvals,)
    test_RectSpline = interpolate.RectBivariateSpline(x, y, fvals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_wireframe(meshxx, meshyy, truef, alpha=0.25, color='C0')
    ax.plot_wireframe(meshxx, meshyy, test_NDBspline(gridxxyy), color='C1')
    ax.plot_wireframe(meshxx, meshyy, test_RectSpline(meshxx, meshyy, grid=False), color='C2')
    plt.show()
