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
    if decimals > 0 and isinstance(value, (float, np.floating)):
        rounded_value = round(value, decimals)
        return ('{:.' + str(decimals) + 'f}').format(rounded_value)
    return str(value)


def to_int_or_float(value) -> int | float:
    """
    Converts a given value into an integer or a floating point value, depending on whether it has decimals or not.

    :param value:   The value to be converted
    :return:        An integer or a floating point value
    """
    value = float(value)
    return int(value) if value % 1 == 0 else value


def format_percentage(fraction: float, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of a percentage using a specific number of decimals.

    :param fraction:    A fraction in [0, 1] to be formatted as a percentage
    :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
    :return:            The textual representation that has been created
    """
    percentage = float(fraction) * 100
    return format_number(percentage, decimals) + '%'  # type: ignore[arg-type]
