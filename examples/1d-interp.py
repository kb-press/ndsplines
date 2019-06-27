"""
===========================
1-Dimensional Interpolation
===========================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm

import itertools

import ndsplines

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

for order in range(4):
    for func in funcs:
        fvals = func(x)
        truef = func(xx)
        plt.figure()
    
        plot_sel = slice(None)
    
        plt.gca().set_prop_cycle(None)
    
        for test_bc in scipy_test_bcs[plot_sel,:]:
            try:
                test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=list(test_bc), k=order)
            except ValueError:
                continue
            else:
                splinef = test_Bspline(xx.copy(), extrapolate=True)
                plt.plot(xx, splinef, '--', lw=3.0, label=str(test_bc) + ' (BSpline)')

        plt.gca().set_prop_cycle(None)
        
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
                plt.plot(xx, NDsplinef, label=', '.join([NDspline_bc_to_string[tuple(bc)] for bc in test_bc]) + ' (ndspline)' )
    
        
        plt.plot(xx, truef, 'k--', label="True " + func.__name__)
        plt.plot(x, fvals, 'ko')
        plt.title('k=%d'%order)
        
        plt.legend(loc='best')
        plt.show()
    
