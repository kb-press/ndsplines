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


scipy_test_bcs = np.array(list(itertools.chain(
    itertools.product(["natural", "clamped", ], repeat=2),
    ((None,None),),
)))

k0_bcs = np.array(list(
itertools.product([(0,0), (0,-1)], repeat=2)
)[:-1])[[-1,0,1]]

NDspline_dict = {"natural": ndsplines.pinned, "clamped": ndsplines.clamped, "not-a-knot": ndsplines.notaknot}

ndsplines_test_bcs = np.array([(NDspline_dict[item[0]], NDspline_dict[item[1]],)  for item in itertools.chain(
    itertools.product(["natural", "clamped", ], repeat=2),
    (("not-a-knot","not-a-knot"),),
    itertools.product(["not-a-knot"],["natural", "clamped", ]),
    itertools.product(["natural", "clamped", ], ["not-a-knot"]),
)])

NDspline_bc_to_string = {tuple(v):k for k,v in NDspline_dict.items()}
NDspline_bc_to_string[(0,-1)] = 'one-sided hold'

for order in range(1,4):
    for func in funcs:
        fvals = func(x)
        truef = func(xx)
        if order > 0:
            fig, axes = plt.subplots(3,1, constrained_layout=True)
        else:
            fig, axes = plt.subplots(2,1, constrained_layout=True)
    
        plot_sel = slice(None)
    
        plt.gca().set_prop_cycle(None)
    
        for test_bc in scipy_test_bcs[plot_sel,:]:
            try:
                test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=list(test_bc), k=order)
            except ValueError:
                continue
            else:
                splinef = test_Bspline(xx.copy(), extrapolate=True)
                axes[0].plot(xx, splinef, '--', lw=3.0, label=str(test_bc) + ' (BSpline)')
                if order > 0:
                    der_Bspline = test_Bspline.derivative()
                    axes[1].plot(xx, der_Bspline(xx.copy()), '--', lw=3.0, label='(BSpline)')
                antider_Bspline = test_Bspline.antiderivative()
                axes[-1].plot(xx, antider_Bspline(xx.copy()), '--', lw=3.0, label='(BSpline)')

        for ax in axes:
            ax.set_prop_cycle(None)
        
        if order == 0:
            bc_iter = k0_bcs
        else:
            bc_iter = ndsplines_test_bcs
        for test_bc in bc_iter[plot_sel,:]:
            try:
                test_NDBspline = ndsplines.make_interp_spline(x, fvals, bcs=test_bc, orders=order)
            except ValueError:
                continue
            else:
                NDsplinef = test_NDBspline(xx.copy())
                axes[0].plot(xx, NDsplinef, label=', '.join([NDspline_bc_to_string[tuple(bc)] for bc in test_bc]) + ' (ndspline)' )
                if order>0:
                    der_NDspline = test_NDBspline.derivative(0)
                    axes[1].plot(xx, der_NDspline(xx.copy()), label='(ndspline)' )
                antider_NDspline = test_NDBspline.antiderivative(0)
                axes[-1].plot(xx, antider_NDspline(xx.copy()), label='(ndspline)')
    
        
        axes[0].plot(xx, truef, 'k--', label="True " + func.__name__)
        axes[0].plot(x, fvals, 'ko')
        plt.suptitle('k=%d'%order)
        
        axes[0].legend(loc='best')
        plt.show()
    
