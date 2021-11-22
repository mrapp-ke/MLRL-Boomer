#!/bin/bash

PYTHON_VERSIONS=("cp37-cp37m" "cp38-cp38" "cp39-cp39")

python -V

for VERSION in "${PYTHON_VERSIONS[@]}"; do
  echo "${VERSION}"
  PYTHON="/opt/python/${VERSION}/bin"
  ln -fs ${PYTHON} /usr/bin/python
  python -V
done
