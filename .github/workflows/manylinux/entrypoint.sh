#!/bin/bash

PYTHON_VERSIONS=$1

for VERSION in "${PYTHON_VERSIONS[@]}"; do
  PYTHON="/opt/python/${VERSION}/bin/python"
  ln -fs ${PYTHON} /usr/bin/python
  make wheel
  . venv/bin/activate
  pip install auditwheel

  for WHEEL in python/subprojects/*/dist/*.whl; do
    auditwheel repair ${WHEEL}
  done

  deactivate
  make clean
done
