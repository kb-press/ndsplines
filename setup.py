import os
from setuptools import setup, Extension

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
                  depends=[os.path.join(name, "{}.h".format(extname)),],
                  optional=True)
    ]
else:
    extensions = []

exec(open('ndsplines/version.py').read())

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'readme.rst'), encoding='utf-8') as f:
    long_description = f.read()

long_description = long_description.replace(
    "https://ndsplines.readthedocs.io/en/latest/",
    "https://ndsplines.readthedocs.io/en/v{}/".format(
        '.'.join(__version__.split('.')[:3])
    )
)

setup(
    name=name,
    version=__version__,
    description="Multi-dimensional splines",
    url="https://github.com/kb-press/ndsplines",
    author="Benjamin Margolis",
    author_email="ben@sixpearls.com",
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    packages=["ndsplines"],
    ext_modules=extensions,
    long_description=long_description,
    license='BSD',
    setup_requires=['numpy'],
    install_requires=['numpy', 'scipy'],
    extras_require={
        'examples': ['matplotlib'],
        'build_ext': ['cython'],
        'docs': ['sphinx', 'sphinx_gallery']
    },
)
