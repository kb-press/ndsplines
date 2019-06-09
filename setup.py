import os
from setuptools import setup, Extension
from distutils.command.sdist import sdist as _sdist

try:
    import numpy
except ImportError:
    use_numpy = False
else:
    use_numpy = True

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

name = "ndsplines"
extname = '_bspl'

if use_cython and use_numpy:
    extensions = cythonize([
        Extension("{}.{}".format(name, extname),
                  [os.path.join(name, "{}.pyx".format(extname))],
                  include_dirs=[numpy.get_include()],
                  depends=[os.path.join(name, "{}.h".format(extname))]),
    ])

elif use_numpy:
    extensions = [
        Extension("{}.{}".format(name, extname),
                  [os.path.join(name, "{}.c".format(extname))],
                  include_dirs=[numpy.get_include()],
                  depends=[os.path.join(name, "{}.h".format(extname))])
    ]
else:
    extensions = []

setup(
    name=name,
    version="0.0.1",
    description="Multi-dimensional splines",
    url="https://github.com/sixpearls/ndsplines",
    author="Benjamin Margolis",
    packages=["ndsplines"],
    ext_modules=extensions,
    license='BSD',
    setup_requires=['numpy'],
    install_requires=['numpy', 'scipy'],
    extras_require={
        'examples': ['matplotlib'],
        'docs': ['sphinx', 'sphinx_gallery'],
    },
)
