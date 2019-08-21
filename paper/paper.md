---
title: 'ndsplines: A Python Library for Tensor-Product B-Splines of Arbitrary Dimension'
tags:
  - Python
  - interpolation
  - spline
authors:
  - name: Benjamin W. L. Margolis
    orcid: 0000-0001-5602-1888
    affiliation: 1
  - name: Kenneth R. Lyons
    orcid: 0000-0002-9143-8459
    affiliation: 1
affiliations:
 - name: University of California, Davis
   index: 1
date: 19 August 2019
bibliography: references.bib
---

# Summary

In order to use data driven models in analysis, the data must be projected onto a set of basis functions. This projection allows the model to be evaluated over a continuous domain. Additionally, the known form of the basis function enables the use of calculus and other symbolic operations to be performed on the representation of the data model. 

B-Splines are a particularly useful basis because they have beneficial analytic properties and because many efficient algorithms for their creation, manipulation, and evaluation have already been developed [@deboor]. Some of the beneficial properties of the B-Spline basis functions include: a closed support, smoothness, and amenability to efficient and stable algorithms. Strictly speaking, the B-Spline basis functions are 1-dimensional maps from the reals to the reals. However, they can be readily extended to N-dimensions by taking tensor products of B-Splines on orthogonal 1-dimensional spaces for each variable. 

B-Splines are well studied in the field of computer graphics where they are particularly used in computer aided-design (CAD) and graphic design programs. Due to this popularity, many programming language ecosystems have implementations of 1- and 2-dimensional splines such as might be used to design curves or surfaces [@scipy.interpolate, @fortran?, @R? @jullia?]. However, the authors are only aware of one general N-dimensional splines implementation [@matlab toolbox].

A general N-dimensional spline implementation is useful for modeling dynamics from data. For example, the American Institute of Aeronautics and Astronautics (AIAA) has proposed a simulation description mark-up language standard. One of the fundamental element is the interpolation of aerodynamic data, which often uses three to five independent variables to capture properties of the airflow and orientation [@daveml].

ndsplines is a Python package for multivariate B-splines with performant NumPy and C (via Cython) implementations [@numpy, @cython]. A primary goal of this package is to provide a unified API for tensor product splines of arbitrary input and output dimension. This implementation is slightly slower than the special-case 1-D and 2-D implementations, but this is a worthwhile trade-off to only use one API for both and any other dimension. Test coverage is nearly 100%, and importantly verifies desirable mathematical properties including the fundamental theorem of calculus holding for any `NDSpline` object.

One author has used this software package and SimuPy [@simupy] to simulate hypersonic entry dynamics in a manner similar to what was described in AIAA’s simulation description mark-up language [@ptero1, @ptero2]. Static stability analysis was also performed using this software package.



# References