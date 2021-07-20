===
API
===

.. rubric:: Creation routines

Routines for creating ``NDSpline`` objects.

.. autosummary::
    :toctree: api

    ~ndsplines.make_lsq_spline
    ~ndsplines.make_interp_spline
    ~ndsplines.make_interp_spline_from_tidy
    ~ndsplines.from_file

.. rubric:: Class methods

Methods of the :doc:`api/ndsplines.NDSpline` class.

.. autosummary::

    ~ndsplines.NDSpline
    ~ndsplines.NDSpline.__call__
    ~ndsplines.NDSpline.derivative
    ~ndsplines.NDSpline.antiderivative
    ~ndsplines.NDSpline.to_file
    ~ndsplines.NDSpline.copy
    ~ndsplines.NDSpline.__eq__
    ~ndsplines.NDSpline.allocate_workspace_arrays
    ~ndsplines.NDSpline.compute_basis_coefficient_selector

.. rubric:: Knots

Utility function for constructing knot arrays.

.. autosummary::
    :toctree: api

    ~ndsplines._not_a_knot

.. rubric:: Implementations

Selection and usage of the Cython or NumPy implementations for B-Spline
evaluation.

.. autosummary::
    :toctree: api

    ~ndsplines.set_impl
    ~ndsplines.get_impl
    ~ndsplines._bspl
    ~ndsplines._npy_bspl
