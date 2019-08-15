"""
===========================
1-Dimensional Interpolation
===========================
"""

import ndsplines
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm
import itertools


def gaussian(x_in):
    z = norm.ppf(.9995)
    x = z*(2*x_in-1)
    return norm.pdf(x)

def sin(x_in):
    x = np.pi*(x_in-0.5)
    return np.sin(x)

def tanh(x_in):
    x = 2*np.pi*(x_in-0.5)
    return np.tanh(x)

funcs = [gaussian, sin, tanh]

x = np.linspace(0, 1, 9)
xx = np.linspace(-.25, 1.25, 1024)
k = 3

for degree in range(0,4):
    for func in funcs:
        fvals = func(x)
        truef = func(xx)
        plt.figure()
    
        plot_sel = slice(None)
    
        plt.gca().set_prop_cycle(None)
        test_Bspline = interpolate.make_interp_spline(x, fvals, k=degree)
        splinef = test_Bspline(xx.copy(), extrapolate=True)
        plt.plot(xx, splinef, '--', lw=3.0, label='scipy.interpolate.make_interp_spline')

        test_NDBspline = ndsplines.make_interp_spline(x, fvals, degrees=degree)
        NDsplinef = test_NDBspline(xx.copy())
        plt.plot(xx, NDsplinef, label='ndspline.make_interp_spline')
        
        plt.plot(xx, truef, 'k--', label="True " + func.__name__)
        plt.plot(x, fvals, 'ko')
        plt.title('k=%d'%degree)
        
        plt.legend(loc='best')
        plt.show()
    
