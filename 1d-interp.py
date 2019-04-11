from scipy import interpolate
import matplotlib.pyplot as plt
import numpy as np
import itertools

import NDBSpline

x = np.r_[-1:-0.5:3j, 0, 0.5:1:3j] *np.pi
fvals = np.sin(x)

k = 3
factor = 1.25
xx = np.r_[-1:1:1024j]*factor*np.pi

test_bcs = np.array(list(itertools.chain(
    itertools.product(["natural", "clamped"], repeat=2),
    ((None,None),),
)))

NDspline_dict = {"natural": NDBSpline.pinned, "clamped": NDBSpline.clamped, None: 0}
skip_size = 1
plt.figure()

plot_sel = [0,-1]


plt.gca().set_prop_cycle(None)

for test_bc in test_bcs[plot_sel,:]:
    test_Bspline = interpolate.make_interp_spline(x, fvals, bc_type=list(test_bc))
    
    splinef = test_Bspline(xx.copy(), extrapolate=True)
    plt.plot(xx[::skip_size], splinef[::skip_size],'--', lw=3.0, label=str(test_bc) + ' (scipy.interpolate)')

plt.gca().set_prop_cycle(None)

for test_bc in test_bcs[plot_sel,:]:
    test_NDBspline = NDBSpline.make_interp_spline(x, fvals, bcs=(NDspline_dict[test_bc[0]], NDspline_dict[test_bc[1]]))
    NDsplienf = test_NDBspline(xx.copy())
    plt.plot(xx, NDsplienf.squeeze(), label=str(test_bc) + ' (ndspline)' )

plt.legend(loc='best')
plt.plot(x, fvals, 'o')
plt.show()
