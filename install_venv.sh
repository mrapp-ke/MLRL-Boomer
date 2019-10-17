#!/bin/bash

# Delete existing virtual environment and re-install it...
rm -rf venv
python3.7 -m venv venv && venv/bin/pip3.7 install python/