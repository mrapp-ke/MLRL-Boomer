#!/bin/bash

PYTHON_VERSIONS=$1
PYTHON_VERSIONS_ARRAY=(${PYTHON_VERSIONS// / })

for VERSION in "${PYTHON_VERSIONS_ARRAY[@]}"; do
  PYTHON="/opt/python/${VERSION}/bin/python"
  [[ -f ${PYTHON} ]] || { echo "Python installation ${PYTHON} does not exist."; exit 1; }
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
      OUT_DIR="wheelhouse/pure/"
      mkdir -p ${OUT_DIR}
      cp -n ${WHEEL} ${OUT_DIR}
    fi
  done

  for WHEEL in wheelhouse/*.whl; do
    OUT_DIR="wheelhouse/${VERSION}/"
    mkdir -p ${OUT_DIR}
    mv $WHEEL ${OUT_DIR}/${WHEEL//-py3-/-${VERSION}-}
  done

  deactivate
  make clean
done
