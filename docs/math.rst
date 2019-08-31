=======
Splines
=======

A general :math:`N` dimensional tensor product B-spline is given by

.. math::

    S(x_1, ..., x_N) = \sum_{i_1=0}^{n_1-1}  \cdots \sum_{i_N=0}^{n_N-1} c_{i_1, ..., i_N} \prod_{j = i}^{N} B_{i_j;k_j,t_j}(x_j)

where :math:`B_{i_j;k_j,t_j}` is the :math:`i_j`-th B-spline basis function of 
degree :math:`k_j` over the knots :math:`{t_j}` for the :math:`j`-th dimension.
Univariate and bivariate splines are special cases of this general form. The 
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


Compact Support
---------------

A key feature of the B-splines is that for :math:`n+k+1` knots, there are
:math:`n` basis functions, but only :math:`k+1` are non-zero on any 
particular span between knots. So to interpolate a large data set
requires only evaluating the linear combination of :math:`k+1` elements (for
each dimension). For example, a 1-dimensional interpolant can be 
evaluated as

.. math::

    S (x) = \sum_{i = \ell}^{\ell+k+1} c_i B_{i;k,t} (x)

where :math:`B_{i;k,t}` is the :math:`i` -th basis function,  :math:`c_i` is
the :math:`i` -th coefficient, and :math:`\ell` is the first index with a 
non-zero basis function at :math:`x`. The sum could easily run over all
:math:`n` elements, but most summands are zero. The evaluation
to be computed very efficiently by performing the minimal number of floating-point
operations.

Basis definition
----------------

Each basis function is defined by a recursive relationship in the order of the 
B-spline. The 0-th degree B-spline has the form

.. math::

    B_{i, 0}(x) = 1, \textrm{if $t_i \le x < t_{i+1}$, otherwise $0$,}

and higher degree B-splines are constructed as

.. math::

    B_{i, k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i, k-1}(x)
                 + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1, k-1}(x)
