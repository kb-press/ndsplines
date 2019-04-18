import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

name = "ndsplines"

extensions = [
    Extension(f"{name}._bspl",
              [os.path.join(name, "_bspl.pyx")],
              include_dirs=[numpy.get_include()],
              depends=[os.path.join(name, "_bspl.h")]),
]

setup(
    name=name,
    version="0.0.1",
    description="Multi-dimensional splines",
    url="https://github.com/sixpearls/ndsplines",
    author="Benjamin Margolis",
    packages=["ndsplines"],
    ext_modules=cythonize(extensions),
    # TODO: figure out how this is supposed to work
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#new-and-changed-setup-keywords
    setup_requires=['Cython', 'numpy'],
    install_requires=['Cython', 'numpy', 'scipy'],
    extras_require={
        'examples': ['matplotlib'],
    },
)
