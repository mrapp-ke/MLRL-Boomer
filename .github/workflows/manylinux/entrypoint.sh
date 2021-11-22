#!/bin/bash

for PYBIN in /opt/python/*/bin; do
  "${PYBIN}/pip" install -r python/requirements.txt
  cd cpp/
  meson setup build/
  cd build/
  meson compile
  meson install
done
