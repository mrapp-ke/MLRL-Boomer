"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path

from setuptools import find_packages, setup

VERSION = (Path(__file__).resolve().parent.parent.parent.parent / '.version').read_text()

setup(extras_require={
    'BOOMER': ['mlrl-boomer==' + VERSION],
    'SECO': ['mlrl-seco==' + VERSION],
},
      packages=find_packages())
