#!/bin/bash

PYTHON_VERSIONS=$1
PYTHON_VERSIONS_ARRAY=(${PYTHON_VERSIONS// / })

for VERSION in "${PYTHON_VERSIONS_ARRAY[@]}"; do
  PYTHON="/opt/python/${VERSION}/bin/python"
  ln -fs ${PYTHON} /usr/bin/python
  make wheel
  . venv/bin/activate
  pip install auditwheel

  for WHEEL in python/subprojects/*/dist/*.whl; do
    LD_LIBRARY_PATH=cpp/build/subprojects/common/ auditwheel repair ${WHEEL}
  done

  deactivate
  make clean
done
