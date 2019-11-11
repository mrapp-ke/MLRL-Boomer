#!/bin/bash

cd python/

# Delete existing files
find -type f -name "*.so" -delete
find -type f -name "*.c" -delete
find -type f -name "*.cpp" -delete

# Compile .pyx files
python setup.py build_ext --inplace
