#!/bin/bash

PYTHON_VERSIONS=$1
PYTHON_VERSIONS_ARRAY=(${PYTHON_VERSIONS// / })

for VERSION in "${PYTHON_VERSIONS_ARRAY[@]}"; do
  PYTHON="/opt/python/${VERSION}/bin/python"
  ln -fs ${PYTHON} /usr/bin/python
  make wheel \
    || { echo "Building wheels failed."; exit 1; }
  . venv/bin/activate
  pip install auditwheel

  for WHEEL in python/subprojects/*/dist/*.whl; do
    if [[ "${WHEEL}" != "python/subprojects/testbed/"* ]]; then
      auditwheel repair ${WHEEL} || { echo "Failed to repair wheel."; auditwheel show ${WHEEL}; exit 1; }
    else
      echo "Keeping wheel ${WHEEL} as it is."
      mkdir -p wheelhouse/pure/
      cp -n ${WHEEL} wheelhouse/pure/
    fi
  done

  for WHEEL in wheelhouse/*.whl; do
    mkdir -p wheelhouse/${VERSION}/
    mv ${WHEEL} wheelhouse/${VERSION}/
  done

  deactivate
  make clean
done
