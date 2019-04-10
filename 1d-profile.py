from scipy import interpolate
import numpy as np
import NDBSpline
from line_profiler import LineProfiler
import timeit

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
    internal_NDBspline = NDBSpline.make_interp_spline(x, fvals,)    
    return internal_NDBspline(xx.copy())

def create_run_test():
    internal_Bspline = interpolate.make_interp_spline(x, fvals)
    internal_NDBspline = NDBSpline.make_interp_spline(x, fvals,)    
    
    internal_Bspline(xx.copy())
    internal_NDBspline(xx.copy())
    
external_Bspline = interpolate.make_interp_spline(x, fvals)
external_NDBspline = NDBSpline.make_interp_spline(x, fvals,)    

def run_scipy():
    return external_Bspline(xx)

def run_ndspline():
    return external_NDBspline(xx)
    
def run1_scipy():
    return external_Bspline(0.0)

def run1_ndspline():
    return external_NDBspline(0.0)

def run_test():
    external_Bspline(0.0) #xx.copy())
    external_NDBspline(0.0) #xx.copy())

run_ndspline()

print("timing external run1")
print("   scipy: ", timeit.timeit(run1_scipy, number=10_000))
print("ndspline: ", timeit.timeit(run1_ndspline, number=10_000))

print("timing external run big")
print("   scipy: ", timeit.timeit(run_scipy, number=1_000))
print("ndspline: ", timeit.timeit(run_ndspline, number=1_000))

lp = LineProfiler()
lp.add_function(NDBSpline.NDBSpline.compute_basis_coefficient_selector)
lp.add_function(NDBSpline.NDBSpline.__call__)
lp_wrapper = lp(external_NDBspline.__call__)
lp_wrapper(0.0)
lp.print_stats()

