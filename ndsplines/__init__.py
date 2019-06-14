# get the main module so we can set impl
from . import ndsplines

# put everything from ndsplines module into package namespace
from .ndsplines import *


def set_impl(name):
    """Set bspl implementation to either cython or numpy."""
    global ndsplines
    if name == 'cython':
        try:
            from . import _bspl
            ndsplines.impl = _bspl
        except ImportError:
            raise ImportError("Can't use cython implementation. Install "
                              "cython then reinstall ndsplines.")
    elif name == 'numpy':
        from . import _npy_bspl
        ndsplines.impl = _npy_bspl


try:
    set_impl('cython')
except ImportError:
    set_impl('numpy')


__all__ = [d for d in dir() if not d.startswith('_')]
