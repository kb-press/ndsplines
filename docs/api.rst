===
API
===

.. currentmodule:: ndsplines

.. rubric:: Creation routines

Routines for creating :class:`NDSpline` objects.

.. autosummary::
    :toctree: api

    make_lsq_spline
    make_interp_spline
    make_interp_spline_from_tidy
    from_file

.. rubric:: Class methods

Methods of the :class:`NDSpline` class.

.. autosummary::
    :toctree: api

    NDSpline
    NDSpline.__call__
    NDSpline.derivative
    NDSpline.antiderivative
    NDSpline.to_file
    NDSpline.copy
    NDSpline.__eq__
    NDSpline.allocate_workspace_arrays
    NDSpline.compute_basis_coefficient_selector

.. rubric:: Knots

Utility function for constructing knot arrays.

.. autosummary::
    :toctree: api

    _not_a_knot

.. rubric:: Implementations

Selection and usage of the Cython or NumPy implementations for B-Spline
evaluation.

.. autosummary::
    :toctree: api

    set_impl
    get_impl
    _bspl
    _npy_bspl
