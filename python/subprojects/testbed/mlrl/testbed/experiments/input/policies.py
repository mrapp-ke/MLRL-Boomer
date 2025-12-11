"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides policies that can be used when dealing with input data.
"""
from enum import StrEnum


class MissingInputPolicy(StrEnum):
    """
    Policies that can be used if an error occurs while reading input data.
    """
    LOG = 'log'
    EXIT = 'exit'
