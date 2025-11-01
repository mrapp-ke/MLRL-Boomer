"""
Author Michael Rapp (michael.rapp.ml@gmail.com)

Provides policies that can be used when dealing with output data.
"""
from enum import StrEnum


class OutputErrorPolicy(StrEnum):
    """
    Policies that can be used if an error occurs while reading output data.
    """
    LOG = 'log'
    EXIT = 'exit'
