

Building Cython Implementation
------------------------------

After profiling revealed that the scipy.interpolate._bspl implementation is 10x
faster, I copied that code over to refactor to make the necessary parts accessible.

Building requires Cython and Numpy::

    $ pip install cython numpy

Now build the ``_bspl`` module::

    $ python setup.py build_ext -i


Installation
------------

Now you can install::

    $ pip install -e .


Documentation
-------------

Build the docs::

    $ pip install -e .[docs]
    $ cd docs
    $ make html
