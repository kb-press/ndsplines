from .ndsplines import *
from .version import __version__

evaluate_spline = None
_impl = ""


def set_impl(name):
    """Set bspl implementation to either cython or numpy.

    Parameters
    ----------
    name : "cython" or "numpy"
        Name of implementation to use.
    """
    global evaluate_spline, _impl
    name = name.lower()
    if name == "cython":
        try:
            from . import _bspl

            ndsplines.impl = _bspl
            evaluate_spline = _bspl.evaluate_spline
            _impl = "cython"
        except ImportError:
            raise ImportError(
                "Can't use cython implementation. Install cython then reinstall "
                "ndsplines."
            )
    elif name == "numpy":
        from . import _npy_bspl

        ndsplines.impl = _npy_bspl
        evaluate_spline = _npy_bspl.evaluate_spline
        _impl = "numpy"
    else:
        raise ValueError("Implementation must be one of {'cython', 'numpy'}")


def get_impl():
    """Get the current bspl implementation as a string."""
    return _impl


try:
    set_impl("cython")
except ImportError:
    set_impl("numpy")


del impl  # imported here to set it but don't need it in the package namespace

__all__ = [d for d in dir() if not d.startswith("_")]
