from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np

x = np.unique(np.r_[-1:1:9j]) * np.pi
fvals = np.sin(x)

coefs = ndimage.spline_filter(fvals)

bspline = interpolate.make_interp_spline(x, fvals)