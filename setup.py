from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([
        Extension("scipy_bspl", 
            ["scipy_bspl.pyx"],
            include_dirs=[numpy.get_include()],
            depends=['scipy_bspl.h']),
    ],)
)
