from scipy import interpolate
import numpy as np
import NDBPoly
from line_profiler import LineProfiler

x = np.r_[-1:1:9j] * np.pi
x = np.r_[-1:-0.5:3j, 0, 0.5:1:3j] *np.pi
fvals = np.sin(x)

k = 3
factor = 1.25
xx = np.r_[-1:1:1024j]*factor*np.pi


NDspline_dict = {"natural": NDBPoly.pinned, "clamped": NDBPoly.clamped, None: 0}


test_Bspline = interpolate.make_interp_spline(x, fvals)
test_NDBspline = NDBPoly.make_interp_spline(x, fvals,)
# test_NDBspline.check_workspace_shapes(xx.shape)
extrap_flag = True


def speed_test():
    splinef = test_Bspline(xx.copy(), extrapolate=extrap_flag)
    NDsplinef = test_NDBspline(xx.copy())


lp = LineProfiler()
# lp.add_function(test_NDBspline.get_us_and_cc_sel)
lp_wrapper = lp(speed_test)
lp_wrapper()
lp.print_stats()

    
speed_test()
