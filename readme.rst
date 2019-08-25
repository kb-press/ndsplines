=========
ndsplines
=========

.. image:: https://dev.azure.com/kb-press/ndsplines/_apis/build/status/kb-press.ndsplines?branchName=master
    :target: https://dev.azure.com/kb-press/ndsplines/_build/latest?definitionId=1&branchName=master
    :alt: Azure Pipelines build status

.. image:: https://readthedocs.org/projects/ndsplines/badge/?version=latest
    :target: https://ndsplines.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation status

.. image:: https://codecov.io/gh/kb-press/ndsplines/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/kb-press/ndsplines
    :alt: Codecov test coverage

This is a Python package for multivariate B-splines with performant NumPy and C
(via Cython) implementations. For a mathematical overview of tensor product 
B-splines, see the Splines_ page of the documentation.

The primary goal of this package is to provide a unified API for tensor product 
splines of arbitrary input and output dimension. For a list of related packages 
see the Comparisons_ page.

Installation
------------

Install ndsplines with pip::

    $ pip install ndsplines

or from source::

    $ git clone https://github.com/kb-press/ndsplines
    $ cd ndsplines
    $ pip install .

Note: In order to use the C implementation, the system must have a C compiler 
configured before installing ndsplines. If installing from source, to use the C
implementation, install with the ``build_ext`` feature (i.e., ``$ pip install 
.[build_ext]``) or install Cython (i.e., ``$ pip install cython``) before 
installing ``ndsplines``.

.. _Splines: https://ndsplines.readthedocs.io/en/latest/math.html
.. _Comparisons: https://ndsplines.readthedocs.io/en/latest/compare.html

Usage
-----

The easiest way to use ``ndsplines`` is to use one of the ``make_*`` 
functions, ``make_interp_spline``, ``make_interp_spline_from_tidy``, or 
``make_lsq_spline``, which return an ``NDSpline`` object which can be used to
evaluate the spline. For example, suppose we have data over a two-dimensional
mesh.

.. code:: python

    import ndsplines
    import numpy as np

    # generate grid of independent variables
    x = np.array([-1, -7/8, -3/4, -1/2, -1/4, -1/8, 0, 1/8, 1/4, 1/2, 3/4, 7/8, 1])*np.pi
    y = np.array([-1, -1/2, 0, 1/2, 1])
    meshx, meshy = np.meshgrid(x, y, indexing='ij')
    gridxy = np.stack((meshx, meshy), axis=-1)

    # evaluate a function to interpolate over input grid
    meshf = np.sin(meshx) * (meshy-3/8)**2 + 2


We can then use ``make_interp_spline`` to create an interpolating spline and
evaluate it over a denser mesh.

.. code:: python

    # create the interpolating splane
    interp = ndsplines.make_interp_spline(gridxy, meshf)

    # generate denser grid of independent variables to interpolate
    sparse_dense = 2**7
    xx = np.concatenate([np.linspace(x[i], x[i+1], sparse_dense) for i in range(x.size-1)]) # np.linspace(x[0], x[-1], x.size*sparse_dense)
    yy = np.concatenate([np.linspace(y[i], y[i+1], sparse_dense) for i in range(y.size-1)]) # np.linspace(y[0], y[-1], y.size*sparse_dense)
    gridxxyy = np.stack(np.meshgrid(xx, yy, indexing='ij'), axis=-1)

    # evaluate spline over denser grid
    meshff = interp(gridxxyy)


Generally, we construct data so that the first `ndim` axes index the independent 
variables and the remaining axes index output. This is a generalization of using
rows to index time and columns to index output variables for We can also create
an interpolating spline frm a tidy data format:

.. code:: python

    tidy_data = np.dstack((gridxy, meshf)).reshape((-1,3))
    tidy_interp = ndsplines.make_interp_spline_from_tidy(tidy_data, 
      [0,1], # columns to use as independent variable data
      [2]) # columns to use as dependent variable data

    print("\nCoefficients all same?", np.all(tidy_interp.coefficients == interp.coefficients))
    print("Knots all same?", np.all([np.all(knot0 == knot1) for knot0, knot1 in zip(tidy_interp.knots, interp.knots)]))

Note however, that the tidy dataset must be over a structured rectangular grid 
equivalent to the N-dimensional representation. Also note that Pandas dataframes
can be used, in which case lists of column names can be used instead of lists of
column indices. 

To see examples for creating least-squares regression splines 
with ``make_lsq_spline``, see the `1D example`_ and `2D example`_. 

Derivatives of constructed splines can be evaluated in two ways. First by using
the `nus` parameter while calling the interpolator or by creating a new spline 
with the ``derivative`` function. In this codeblock, we show both methods of 
evaluating derivatives in each direction.

.. code:: python

    # two ways to evaluate derivatives x-direction: create a derivative spline or call with nus:
    deriv_interp = interp.derivative(0)
    deriv1 = deriv_interp(gridxxy)
    deriv2 = interp(gridxy, nus=np.array([1,0]))

    # two ways to evaluate derivative - y direction
    deriv_interp = interp.derivative(1)
    deriv1 = deriv_interp(gridxy)
    deriv2 = interp(gridxxyy, nus=np.array([0,1]))

The ``NDSpline`` class also has an ``antiderivative`` method for creating a 
spline representative of the anti-derivative in the specified direction.

.. code:: python

    # Calculus demonstration
    interp1 = deriv_interp.antiderivative(0)
    coeff_diff = interp1.coefficients - interp.coefficients
    print("\nAntiderivative of derivative:\n","Coefficients differ by constant?", np.allclose(interp1.coefficients+2.0, interp.coefficients))
    print("Knots all same?", np.all([np.all(knot0 == knot1) for knot0, knot1 in zip(interp1.knots, interp.knots)]))

    antideriv_interp = interp.antiderivative(0)
    interp2 = antideriv_interp.derivative(0)
    print("\nDerivative of antiderivative:\n","Coefficients the same?", np.allclose(interp2.coefficients, interp.coefficients))
    print("Knots all same?", np.all([np.all(knot0 == knot1) for knot0, knot1 in zip(interp2.knots, interp.knots)]))


.. _1D example : https://ndsplines.readthedocs.io/en/latest/auto_examples/1d-lsq.html
.. _2D example: https://ndsplines.readthedocs.io/en/latest/auto_examples/2d-lsq.html


Contributing
------------

Please feel free to share any thoughts or opinions about the design and
implementation of this software by `opening an issue on GitHub
<https://github.com/kb-press/ndsplines/issues/new>`_. Constructive feedback is
welcomed and appreciated.

Bug fix pull requests are always welcome. For feature additions, breaking 
changes, etc. check if there is an open issue discussing the change and 
reference it in the pull request. If there isn't one, it is recommended to open 
one with your rationale for the change before spending significant time 
preparing the pull request.

Ideally, new/changed functionality should come with tests and documentation. If
you are new to contributing, it is perfectly fine to open a work-in-progress
pull request and have it iteratively reviewed.

Testing
=======

To test, install the developer requirements and use ``pytest``::

    $ pip install -r requirements-dev.txt
    $ pip install -e .
    $ pytest

Documentation
=============

To build the docs, install the ``docs`` feature requirements (a subset of
the developer requirements above)::

    $ pip install -e .[docs]
    $ cd docs
    $ make html

