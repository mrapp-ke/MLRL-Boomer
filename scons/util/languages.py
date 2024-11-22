"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that help to distinguish between different programming languages.
"""
from enum import Enum


class Language(Enum):
    """
    Different programming languages.
    """
    CPP = {'cpp', 'hpp'}
    PYTHON = {'py'}
    CYTHON = {'pyx', 'pxd'}
    MARKDOWN = {'md'}
    YAML = {'yaml', 'yml'}
