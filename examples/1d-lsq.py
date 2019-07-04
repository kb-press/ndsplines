"""
===============================
1-Dimensional Least Squares Fit
===============================
"""

import ndsplines
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)

t = [-1, 0, 1]
k = 3
t = np.r_[(x[0],)*(k+1),
          t,
          (x[-1],)*(k+1)]

ndspl = ndsplines.make_lsq_spline(x[:, None], y[:, None], [t], np.array([k]))
ispl = interpolate.make_lsq_spline(x, y, t, k)

xs = np.linspace(-3, 3, 100)
plt.figure()
plt.plot(x, y, 'o', ms=5)
plt.plot(xs, ndspl(xs).squeeze(), label='LSQ ND spline')
plt.plot(xs, ispl(xs), '--', label='LSQ scipy.interpolate spline')
plt.legend(loc='best')
plt.show()

print("Computed coefficients close?", np.allclose(ndspl.coefficients, ispl.c))
