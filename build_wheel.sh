#!/bin/bash

# Script to build manylinux2010_x86_64 wheels.

# This is intended for use with the quay.io/pypa/manylinux2010_x86_64 Docker
# image. Azure Pipelines is set up to run this in such a container with the
# working directory mounted at /io.

set -e -x

platform="manylinux2010_x86_64"

for pybin in /opt/python/cp3*/bin; do
    $pybin/pip install -r requirements.txt
    $pybin/python setup.py bdist_wheel
done

for whl in dist/*.whl; do
    auditwheel repair $whl --plat $platform -w dist
done
