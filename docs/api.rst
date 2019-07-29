===
API
===

.. rubric:: Class methods 

Methods of the :doc:`api/ndsplines.NDSpline` class.

.. autosummary::
    
    ~ndsplines.NDSpline.__init__
    ~ndsplines.NDSpline.allocate_workspace_arrays
    ~ndsplines.NDSpline.compute_basis_coefficient_selector
    ~ndsplines.NDSpline.__call__
    ~ndsplines.NDSpline.derivative
    ~ndsplines.NDSpline.antiderivative
    ~ndsplines.NDSpline.to_file
   

.. rubric:: Creation routines

Routines for creating ``NDSpline`` objects.

.. autosummary::
    :toctree: api
    
    ~ndsplines.make_lsq_spline
    ~ndsplines.make_interp_spline
    ~ndsplines.make_interp_spline_from_tidy
    ~ndsplines.from_file

.. rubric:: Implementations

Select and use the Cython or NumPy impelementations for B-Spline evaluation.

.. autosummary::
    :toctree: api

    ~ndsplines.set_impl
    ~ndsplines._bspl
    ~ndsplines._npy_bspl