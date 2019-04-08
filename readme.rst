Splines
-------

The basic idea of (tensor product) splines is that (in each dimension) you have 
a $n$ (or $n_i$, with i indicating the dimension) samples of un-evenly spaced 
data $(t_i, y_i=f(t_i))$ (so multi-dimensional data can be rectangular but not 
unstructured). A B-spline is a piecewise $k$-th order polynomial interpolant 
that can be constructed with nice properties such as $C^{k-1}$ continuity. The 
interpolant is represented by $n+k+1$ knots, which are essentially the $t_i$ but
extra points that are determined by the data independent variable and the 
desired properties at the ends of the interpolant including extrapolation 
properties and $n$ coefficients which are the projection of the $y_i$ data onto
the B-spline basis functions.

Intuition on order: A $k=0$ B-spline is piecewise constant, often called 
"nearest-neighbor" interpolation. A $k=0$ B-spline is piecewise linear. A $k=3$
B-spline is the default and produces aesthetically pleasing curves that can also
be represented as the solution to thin-beam bending dynamics.

A key feature of the B-splines is that for $n$ data-points, there are $n$
basis functions, but only $k+1$ are non-zero on any particular panel (domain
between knots). So to interpolate a large data set requires only evaluating
the linear combination of $k+1$ elements (for each dimension), as in

\hat{f} (x) = \sum_{i = j}^{j+k+1} c_i u_i (x)

where $u_i$ is the $i$th basis function, and $c_i$ is the ith coefficient, and
$j$ is the first index with a non-zero basis function. The sum could easily run
over all $n$ elemnets, but most summands are zero.

In Matlab, these spline interpolations are implemnted in:
    - Spline Toolbox, which also includes regression using splines
    - `interpn`, `interp{1|2|3}`, and `spline` (an alias for interp1) which 
      together may be a replacement for the spline toolbox?

 Spline Toolbox was written by Carl de Boor, who wrote many seminal articles 
 and books on splines. Most algorithms for evaluating the basis functions and
 computing the coefficients and knots from data were developed by de Boor.


In Python, SciPy provides:

    - `splrep` and `splev`, which are thin wrappers around fortran functions by
      the same name. These functions construct the knot and coefficient sequence
      given a sequence of 1-D data and evaluate the B-splines given the knot and
      coefficient sequence. The fortran supports only a single set of 
      coefficients, but the python wrapper will recursively call itself to 
      evaluate using multiple coefficients.

    - I believe there was short-lived class-based interface for `splrep` and 
      `splev`.

    - `BSpline`, which is a newer interface using pure Python, Cython and C for
      1-D splines. This supports a Cython-loop over sets of coefficients.

    - A number of interfaces for N-D unevenly spaced data limited to $k=0$ 
      (nearest neighbor) or $k=1$ (piecewise linear) interpolation (`interpn` 
      and `RegularGridInterpolator`), 2D tensor spline interpolation
      (`RectBivariateSpline`), or N-D evenly spaced data (`ndimage`).


So far, I have worked on 2 implementations:
    - Using splev on a set of coefficients to evaluate the u_i for each 
      dimension, then collect the evaluation and use einsum to multiply and
      sum the coefficients. This is definitely slow because I am allocating 
      memory and iterating over a bunch of known zeros. I didn't realize that
      to evaluate each basis function, I am looping in python over the the 
      associated $n$ (for that dimension)
    - Before I reviewed my math, I thought `ndimage` could be used with a simple
      transformation for non-uniform knots. 

Speed up thoughts:
    - Drop-in Bspline for `splev` to reduce costs associated with evaluating
      the coefficients
    - Pure Python/NumPy implementation to efficiently evaluate the required
      spline basis functions and perform the inner product
    - New Cython-based implementation modeled after BSpline code

For the first two, pre-allocating and keeping intermediate arrays between calls
(check for same shape of `x`) would probably further speed-up. Between the 
second two, I think I have a 


It was easy to use the 1-D interfaces of knot/coefficient construction to build
N-D knots/coefficients. It would be nice to support N-D least squares regression
but I am not familiar with the direct construction of the linear equation.

Assuming I effectively use the C(ython) API for NumPy, the Cython should be
strictly faster, but the Pure Python/NumPy implementation should approach that
speed as dimension, dataset size, and evaluation points increase.


Profiling
---------

built into ipython:

From IPython shell
In [ ]: %run -p 2d-profile.py

There's also a python built-in that I don't know how to read the binary output of:

python -m cProfile -o 2d.prof 2d-profile.py

Shockingly, the first hit on google is just a 3rd party package for reading it 
because there isn't/wasn't a good way to read them. Okay. Using line_profiler
package:

$ pip install line_profiler
$ python 1d-profile.py


I made the profiling explicit and decoupled from actual source code.

Building
--------
After profiling revealed that the scipy.interpolate._bspl implementation is 10x
faster, I copied that code over to refactor to make the necessary parts accessible.
There may be other ways to build, but from the directory

$ python setup.py build_ext -i

definitely builds it and makes it importable, but I'm not sure if it's the only/best
way.
