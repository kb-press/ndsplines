import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

numpy_only = os.environ.get("NDSPLINES_NUMPY_ONLY") == "1"

ext_modules = []

if not numpy_only:
    ext_modules.extend(
        cythonize(
            Extension(
                "ndsplines._bspl",
                [os.path.join("ndsplines", "_bspl.pyx")],
                include_dirs=["ndsplines", numpy.get_include()],
            )
        )
    )

setup(ext_modules=ext_modules)
