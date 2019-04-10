import NDBSpline
import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * np.random.randn(50)


t = [-1, 0, 1]
k = 3
t = np.r_[(x[0],)*(k+1),
          t,
          (x[-1],)*(k+1)]

spl = NDBSpline.make_lsq_spline(x[None, :], y[None, :], [t], np.array(k))


xs = np.linspace(-3, 3, 100)
plt.plot(x, y, 'o', ms=5)
plt.plot(xs, spl(xs), label='LSQ spline')
plt.legend(loc='best')
plt.show()
