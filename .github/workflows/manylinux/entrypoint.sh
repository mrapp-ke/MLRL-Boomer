#!/bin/bash

PYTHON_VERSIONS="cp37-cp37m cp38-cp38 cp39-cp39"

for VERSION in ${PYTHON_VERSION}; do
  PYTHON="/opt/python/${VERSION}/bin"
  ${PYTHON} -m venv venv
done
