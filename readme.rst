=========
ndsplines
=========

This is a Python package for multivariate B-splines with performant C(ython) and
NumPy implementations.

Usage
-----

The easiest way to use ndspliens is to call ``make_interp_spline`` or 
``make_lsq_spline`` on your problem data, as in

.. code:: python

     import ndsplines
     import matplotlib.pyplot as plt
     import numpy as np

     x = np.linspace(-3, 3, 50)
     y = np.exp(-x**2) + 0.1 * np.random.randn(50)

     t = [-1, 0, 1]
     k = 3
     t = np.r_[(x[0],)*(k+1),
               t,
               (x[-1],)*(k+1)]

     ndspl = ndsplines.make_lsq_spline(x, y, t, 3)

     xs = np.linspace(-3, 3, 100)
     plt.figure()
     plt.plot(x, y, 'o', ms=5)
     plt.plot(xs, ndspl(xs).squeeze(), label='LSQ spline')
     plt.legend(loc='best')
     plt.show()




Installation
------------

Install ndsplines with pip::

    $ pip install ndsplines

or from source::

    $ git clone https://github.com/sixpearls/ndsplines
    $ cd ndsplines
    $ pip install -e .

Note that in order to use the C(ython) implementation, NumPy must be installed
and a C compiler configured before installing ndsplines.


Documentation
-------------

Build the docs::

    $ pip install -e .[docs]
    $ cd docs
    $ make html


