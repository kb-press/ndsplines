# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import ndsplines

# -- Project information -----------------------------------------------------

project = 'ndsplines'
copyright = '2019, Benjamin Margolis'
author = 'Benjamin Margolis'

# The full version, including alpha/beta/rc tags
release = ndsplines.__version__
version = ndsplines.__version__


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
    # can enable this later if we want
    'sphinx.ext.autosummary',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# sphinx-gallery configuration
# see https://sphinx-gallery.github.io/configuration.html
sphinx_gallery_conf = {
    'examples_dirs': ['../examples', '../benchmarks'],
    'gallery_dirs': ['auto_examples', 'auto_benchmarks'],
    'filename_pattern': r'.*\.py',
    'line_numbers': True,
    'download_all_examples': False,
}


autosummary_generate = True
