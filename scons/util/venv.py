"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with virtual Python environments.
"""
import sys


def in_virtual_environment() -> bool:
    """
    Returns whether the current process is executed in a virtual environment or not.

    :return: True, if the current process is executed in a virtual environment, False otherwise
    """
    return sys.prefix != sys.base_prefix
