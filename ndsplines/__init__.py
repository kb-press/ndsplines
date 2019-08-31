from .ndsplines import *
from .version import __version__

evaluate_spline = None

def set_impl(name):
    """Set bspl implementation to either cython or numpy.

    Parameters
    ----------
    name : 'cython' or 'numpy'
        Name of implementation to set to.

    """
    global evaluate_spline
    if name == 'cython':
        try:
            from . import _bspl
            ndsplines.impl = _bspl
            evaluate_spline = _bspl.evaluate_spline
        except ImportError:
            raise ImportError("Can't use cython implementation. Install "
                              "cython then reinstall ndsplines.")
    elif name == 'numpy':
        from . import _npy_bspl
        ndsplines.impl = _npy_bspl
        evaluate_spline = _npy_bspl.evaluate_spline


try:
    set_impl('cython')
except ImportError:
    set_impl('numpy')


__all__ = [d for d in dir() if not d.startswith('_')]
