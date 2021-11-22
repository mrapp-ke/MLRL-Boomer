#!/bin/bash

PYTHON_VERSIONS=("cp37-cp37m" "cp38-cp38" "cp39-cp39")

for VERSION in "${PYTHON_VERSIONS[@]}"; do
  PYTHON="/opt/python/${VERSION}/bin/python"
  ln -fs ${PYTHON} /usr/bin/python
  make venv
done
