import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm

import itertools

import ndsplines

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

funcs = [gaussian, sin, tanh]

x = np.linspace(0, 1, 7)
xx = np.linspace(-.25, 1.25, 1024)
k = 3


test_bcs = np.array(list(itertools.chain(
    itertools.product(["natural", "clamped"], repeat=2),
    ((None,None),),
)))

NDspline_dict = {"natural": ndsplines.pinned, "clamped": ndsplines.clamped, None: 0}

for func in funcs:
    fvals = func(x)
    truef = func(xx)
    plt.figure()

    plot_sel = slice(None)

    plt.gca().set_prop_cycle(None)

    for test_bc in test_bcs[plot_sel,:]:
        test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=list(test_bc))
        
        splinef = test_Bspline(xx.copy(), extrapolate=True)
        plt.plot(xx, splinef,'--', lw=3.0, label=str(test_bc) + ' (BSpline)')

    plt.gca().set_prop_cycle(None)

    for test_bc in test_bcs[plot_sel,:]:
        test_NDBspline = ndsplines.make_interp_spline(x, fvals, bcs=(NDspline_dict[test_bc[0]], NDspline_dict[test_bc[1]]))
        NDsplienf = test_NDBspline(xx.copy())
        plt.plot(xx, NDsplienf[0], label=str(test_bc) + ' (ndspline)' )

    
    plt.plot(xx, truef, 'k--', label="True " + func.__name__)
    
    plt.legend(loc='best')
    plt.plot(x, fvals, 'o')
    plt.show()
    