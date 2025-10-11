"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
from numbers import Number
from typing import Any

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


def parse_number(value: Any, percentage: bool = False) -> Number:
    """
    Parses a given value and converts it into a number. If the value cannot be parsed, a `ValueError` is raised.

    :param value:       The value to be parsed
    :param percentage:  True, if the given value is a percentage, False otherwise. Percentages will be converted into
                        values in [0, 1]
    :return:            A number
    """
    value = float(str(value))

    if value % 1 == 0:
        value = int(value)

    return value / 100 if percentage else value


def format_percentage(fraction: float, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of a percentage using a specific number of decimals.

    :param fraction:    A fraction in [0, 1] to be formatted as a percentage
    :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
    :return:            The textual representation that has been created
    """
    percentage = float(fraction) * 100
    return format_number(percentage, decimals) + '%'  # type: ignore[arg-type]
