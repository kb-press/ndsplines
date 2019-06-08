import os
from setuptools import setup, Extension
from distutils.command.sdist import sdist as _sdist
import numpy

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

name = "ndsplines"

cmdclass = {}

if use_cython:
    extensions = cythonize([
        Extension(f"{name}._bspl",
                  [os.path.join(name, "_bspl.pyx")],
                  include_dirs=[numpy.get_include()],
                  depends=[os.path.join(name, "_bspl.h")]),
    ])

    class sdist(_sdist):
        def run(self):
            cythonize(['cython/mycythonmodule.pyx'])
            _sdist.run(self)
    cmdclass['sdist'] = sdist
else:
    extensions = [
        Extension(f"{name}._bspl",
                  [os.path.join(name, "_bspl.c")],
                  include_dirs=[numpy.get_include()],
                  depends=[os.path.join(name, "_bspl.h")])
    ]

setup(
    name=name,
    version="0.0.1",
    description="Multi-dimensional splines",
    url="https://github.com/sixpearls/ndsplines",
    author="Benjamin Margolis",
    packages=["ndsplines"],
    cmdclass=cmdclass,
    ext_modules=extensions,
    license='BSD',
    setup_requires=['numpy'],
    install_requires=['numpy', 'scipy'],
    extras_require={
        'examples': ['matplotlib'],
        'docs': ['sphinx', 'sphinx_gallery'],
    },
)
