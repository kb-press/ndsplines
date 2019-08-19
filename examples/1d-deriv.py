"""
===========================
1-Dimensional Derivatives
===========================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm

import itertools

import ndsplines


def sin(x_in):
    x = np.pi*(x_in-0.5)
    return np.sin(x)

def cos(x_in):
    x = np.pi*(x_in-0.5)
    return np.cos(x)

funcs = [sin, cos]

x = np.linspace(0, 1, 9)
xx = np.linspace(-.0625, 1.0625, 1024)
k = 3

for degree in range(1,4):
    for func in funcs:
        fvals = func(x)
        truef = func(xx)
        if degree > 0:
            fig, axes = plt.subplots(3,1, constrained_layout=True)
        else:
            fig, axes = plt.subplots(2,1, constrained_layout=True)
    
        plot_sel = slice(None)
    
        plt.gca().set_prop_cycle(None)
        test_Bspline = interpolate.make_interp_spline(x, fvals, k=degree)
        splinef = test_Bspline(xx.copy(), extrapolate=True)
        axes[0].plot(xx, splinef, '--', lw=3.0, label='BSpline')
        if degree > 0:
            der_Bspline = test_Bspline.derivative()
            axes[1].plot(xx, der_Bspline(xx.copy()), '--', lw=3.0, label='BSpline')
        antider_Bspline = test_Bspline.antiderivative()
        axes[-1].plot(xx, antider_Bspline(xx.copy()), '--', lw=3.0, label='BSpline')

        for ax in axes:
            ax.set_prop_cycle(None)
        test_NDBspline = ndsplines.make_interp_spline(x, fvals, degrees=degree)

        NDsplinef = test_NDBspline(xx.copy())
        axes[0].plot(xx, NDsplinef, label='ndspline' )
        if degree>0:
            der_NDspline = test_NDBspline.derivative(0)
            axes[1].plot(xx, der_NDspline(xx.copy()), label='ndspline' )
        antider_NDspline = test_NDBspline.antiderivative(0)
        axes[-1].plot(xx, antider_NDspline(xx.copy()), label='ndspline')

        
        axes[0].plot(xx, truef, 'k--', label="True " + func.__name__)
        axes[0].plot(x, fvals, 'ko')
        plt.suptitle('k=%d'%degree)
        
        axes[0].legend(loc='best')
        plt.show()
    
