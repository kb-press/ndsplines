==========
Comparison
==========

Here are some notes on spline functionality provided by other software.

In Python, SciPy's `interpolate
<https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_ module provides:

    - A number of wrappers around the fortran ``fitpack`` library by P. Dierckx:

      - ``splrep`` and ``splev``, which are thin wrappers around fortran functions by
        the same name. These functions construct the knot and coefficient sequence
        given a sequence of 1-D data and evaluate the B-splines given the knot and
        coefficient sequence. The fortran supports only a single set of 
        coefficients, but the python wrapper will recursively call itself to 
        evaluate using multiple coefficients. There are a number of other 
        functional wrappers around univariate spline functions from ``fitpack``.

      - There are also class-based wrappers for ``fitpack``'s univariate splines: 
        ``UnivariateSpline``, ``InterpolatedUnivariateSpline``, and 
        ``LSQUnivariateSpline``.

      - Functional interface to ``bisplrep`` and ``bisplev``. Note that ``bisplrep``
        can construct a B-spline that interpolates data on an unstructured
        grid.

      - Class-based interfaces for ``fitpack``'s bivariate splines, 
        ``RectBivariateSpline``, ``RectSphereBivariateSpline``, 
        ``SmoothBivariateSpline``, ``SmoothSphereBivariateSpline``, 
        ``LSQBivariateSpline``, ``LSQSphereBivariateSpline``.

      - An interface to ``fitpack``'s bivariate splines is also provided in 
        ``interp2d``.

    - ``BSpline``, which is a new implementation using Cython and C for
      1-D splines. This supports a Cython-loop over sets of coefficients. This
      is also wrapped by ``interp1d``. 

    - ``BPoly`` is a pure Python implementation for 1-D splines.

    - A number of interfaces for N-D unevenly spaced data limited to ``k=0``
      (nearest neighbor) or ``k=1`` (piecewise linear) interpolation:

        - ``RegularGridInterpolator`` has a pure python implementation for
          linear and nearest-neighbor interpolation

        - ``griddata`` which wraps Cython implmentations ``LinearNDInterpolator``   
          and ``NearestNDInterpolator``

        - ``interpn`` which uses ``RegularGridInterpolator`` for N-D or 
          ``RectBivariateSpline`` for 2-D interpolation.

    - ``ndimage``, which provides N-D interpolation of any order if the spacing
      in each dimension is uniform.

There are also other interpolation methods using other basis functions.

In Matlab, these spline interpolations are implemented in:
    - Spline Toolbox, which also includes regression using splines
    - ``interpn``, ``interp{1|2|3}``, and ``spline`` (an alias for interp1) which 
      together may be a replacement for the spline toolbox.

 `Spline Toolbox` was written by Carl de Boor, who wrote many seminal articles 
 and books on splines. Many commonly used algorithms for evaluating the basis 
 functions and computing the coefficients and knots from data were developed 
 by de Boor.
