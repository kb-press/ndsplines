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


def create_run_scipy():
    internal_Bspline = interpolate.make_interp_spline(x, fvals)
    return internal_Bspline(xx.copy())

def create_run_ndspline():
    internal_NDBspline = NDBPoly.make_interp_spline(x, fvals,)    
    return internal_NDBspline(xx.copy())

def create_run_test():
    internal_Bspline = interpolate.make_interp_spline(x, fvals)
    internal_NDBspline = NDBPoly.make_interp_spline(x, fvals,)    
    
    internal_Bspline(xx.copy())
    internal_NDBspline(xx.copy())
    
external_Bspline = interpolate.make_interp_spline(x, fvals)
external_NDBspline = NDBPoly.make_interp_spline(x, fvals,)    

def run_scipy():
    return external_Bspline(xx.copy())

def run_ndspline():
    return external_NDBspline(xx.copy())

def run_test():
    external_Bspline(xx.copy())
    external_NDBspline(xx.copy())



lp = LineProfiler()
# lp.add_function(NDBPoly.NDBPoly.get_us_and_cc_sel)

lp_wrapper = lp(run_test)
lp_wrapper()
lp.print_stats()

