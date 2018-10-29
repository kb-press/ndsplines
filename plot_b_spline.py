from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import numpy as np

# ndimage.spline_filter1d()

xmax = 5
x = np.r_[-2*xmax:2*xmax:1024j].reshape(1,-1)
y = np.r_[0, 1, 0:0:(xmax//1-1)*1j]

out = ndimage.map_coordinates(y, x, prefilter=False, order=2, mode='nearest')

plt.figure()
plt.plot(x.squeeze(), out.squeeze())
plt.show()