"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
from numbers import Number

import numpy as np

OPTION_DECIMALS = 'decimals'

OPTION_PERCENTAGE = 'percentage'


def format_number(value: Number, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of a value using a specific number of decimals, if the value is a
    floating point value.

    :param value:       The value
    :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
    :return:            The textual representation that has been created
    """
    if isinstance(value, (int, np.integer)):
        return str(value)
    return ('{:.' + str(decimals) + 'f}').format(round(value, decimals)) if decimals > 0 else str(value)


def format_percentage(fraction: float, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of a percentage using a specific number of decimals.

    :param fraction:    A fraction in [0, 1] to be formatted as a percentage
    :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
    :return:            The textual representation that has been created
    """
    return format_number(float(fraction) * 100, decimals) + '%'
