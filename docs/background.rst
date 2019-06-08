Splines
-------

A general :math:`N` dimensional tensor product B-spline is given by

.. math::

    S(x_1, ..., x_N) = \sum_{i_1=0}^{n_1-1}  \cdots \sum_{i_N=0}^{n_N-1} c_{i_1, ..., i_N} \prod_{j = i}^{N} B_{i_j;k_j,t_j}(x_j)

where :math:`B_{i_j;k_j,t_j}` is the :math:`i_j`-th B-spline basis function of 
degree :math:`k_j` over the knots :math:`{t_j}` for the :math:`j`-th dimension.
Univariate and bivariate splines are special cases of the general form. The 
usual form for a univariate B-spline is

.. math::

    S(x) = \sum_{i=0}^{n-1} c_i B_{i;k,t} (x)

which takes the generalized form by letting :math:`N=1`, and indexing :math:`i`,
:math:`k`, :math:`t`, :math:`n`, and :math:`x` by a subscript 1, for the first and only
dimension. Similarly, the usual form for a bivariate spline is


.. math::

    S(x, y) = \sum_{i=0}^{n_x -1} \sum_{j=0}^{n_y -1} c_{i,j} B_{i; k_x, t_x} (x) B_{j; k_y, t_y} (y)

which takes the generalized form if :math:`x=x_1`, :math:`y=x_2`, :math:`i = i_1`, 
:math:`j=i_2`, and  using :math:`x` and :math:`y` subscripts to indicate the
data for each dimension.

Order of B-Spline
-----------------

A :math:`k=0` degree B-spline is piecewise constant, which can be used for
"nearest-neighbor" interpolation. A :math:`k=1` degree B-spline is piecewise 
linear. A :math:`k=3` degree B-spline is the default and produces aesthetically 
pleasing curves that can also be represented as the solution to thin-beam 
bending mechanics.

For an :math:`N`-dimensional tensor-product B-spline, each dimension can have
its own order.


Notes on Implementation
-----------------------

A key feature of the B-splines is that for :math:`n+k+1` knots, there are
:math:`n` basis functions, but only :math:`k+1` are non-zero on any 
particular panel (domain between knots). So to interpolate a large data set
requires only evaluating the linear combination of :math:`k+1` elements (for
each dimension). So, for example, a 1-dimensional interpolant can be 
evaluated as

.. math::

    S (x) = \sum_{i = \ell}^{\ell+k+1} c_i B_{i;k,t} (x)

where :math:`B_{i;k,t}` is the :math:`i` -th basis function,  :math:`c_i` is
the :math:`i` -th coefficient, and :math:`\ell` is the first index with a 
non-zero basis function at :math:`x`. The sum could easily run over all
:math:`n` elements, but most summands are zero. This allows the the evaluation
to be computed very efficiently.

Each basis function is defined by a recursive relationship in the order of the 
B-spline. The 0-th degree B-spline has the form

.. math::

    B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}

and higher degree B-splines are constructed as

.. math::

    B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)



Other implementations
---------------------

In Matlab, these spline interpolations are implemnted in:
    - Spline Toolbox, which also includes regression using splines
    - `interpn`, `interp{1|2|3}`, and `spline` (an alias for interp1) which 
      together may be a replacement for the spline toolbox?

 Spline Toolbox was written by Carl de Boor, who wrote many seminal articles 
 and books on splines. Most algorithms for evaluating the basis functions and
 computing the coefficients and knots from data were developed by de Boor.


In Python, SciPy provides:

    - A number of wrappers around the fortran `fitpack` library by P. Dierckx:

      - `splrep` and `splev`, which are thin wrappers around fortran functions by
        the same name. These functions construct the knot and coefficient sequence
        given a sequence of 1-D data and evaluate the B-splines given the knot and
        coefficient sequence. The fortran supports only a single set of 
        coefficients, but the python wrapper will recursively call itself to 
        evaluate using multiple coefficients. There are a number of other 
        functional wrappers around univariate spline functions from `fitpack`.

      - There are also class-based wrappers for `fitpack`'s univariate splines: 
        `UnivariateSpline`, `InterpolatedUnivariateSpline`, and 
        `LSQUnivariateSpline`

      - Functional interface to `bisplrep` and `bisplev`. Note that `bisplrep`
        can construct a B-spline that interpolates data on an unstructured
        grid.
        TODO: I believe this functionality can be easily re-produced using the
        least squares constructor; we should make sure we can do this.

      - Class-based interfaces for `fitpack`'s bivariate splines, 
        `RectBivariateSpline`, `RectSphereBivariateSpline`, 
        `SmoothBivariateSpline`, `SmoothSphereBivariateSpline`, 
        `LSQBivariateSpline`, `LSQSphereBivariateSpline`

      - An interface to `fitpack`'s bivariate splines is also provided in 
        `interp2d`.

    - `BSpline`, which is a new implementation using Cython and C for
      1-D splines. This supports a Cython-loop over sets of coefficients. This
      is also wrapped by `interp1d`. 

    - `BPoly` is a pure Python implementation for 1-D splines.

    - A number of interfaces for N-D unevenly spaced data limited to `k=0`
      (nearest neighbor) or `k=1` (piecewise linear) interpolation:

        - `RegularGridInterpolator` has a pure python implementation for
          linear and nearest-neighbor interpolation

        - `griddata` which wraps Cython implmentations `LinearNDInterpolator`   
          and `NearestNDInterpolator`

        - `interpn` which uses `RegularGridInterpolator` for N-D or 
          `RectBivariateSpline` for 2-D interpolation.

    - `ndimage`, which provides N-D interpolation of any order if the spacing
      in each dimension is uniform.

There are also other interpolation methods using other basis functions.