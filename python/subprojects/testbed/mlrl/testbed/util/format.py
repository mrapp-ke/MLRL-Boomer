"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""

from numbers import Number
from typing import Any

from mlrl.util.format import format_value

OPTION_DECIMALS = 'decimals'

OPTION_PERCENTAGE = 'percentage'


def parse_value(value: Any) -> Any:
    """
    Parses a given value and converts it into a number, if possible.

    :param value:   The value to be parsed
    :return:        The given value or a number
    """
    try:
        return to_int_or_float(value)
    except ValueError:
        return value


def parse_number(value: Any, percentage: bool = False) -> Number:
    """
    Parses a given value and converts it into a number. If the value cannot be parsed, a `ValueError` is raised.

    :param value:       The value to be parsed
    :param percentage:  True, if the given value is a percentage, False otherwise. Percentages will be converted into
                        values in [0, 1]
    :return:            A number
    """
    value = to_int_or_float(value)
    return value / 100 if percentage else value


def to_int_or_float(value: Any) -> int | float:
    """
    Converts a given value into an integer or a floating point value, depending on whether it has decimals or not.

    :param value:   The value to be converted
    :return:        An integer or a floating point value
    """
    if isinstance(value, int):
        return value

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
    return f'{format_value(percentage, decimals)}%'
