"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""

from numbers import Number
from typing import Any

from mlrl.util.format import format_value
from mlrl.util.options import Options

OPTION_DECIMALS = 'decimals'

OPTION_PERCENTAGE = 'percentage'

OPTION_THOUSANDS_SEPARATORS = 'thousands_separators'


def get_format_kwargs(default_options: Options | None = None, **kwargs) -> dict[str, Any]:
    """
    Returns the keyword arguments to be passed to the function `mlrl.util.format.format_value`. Default values can be
    overwritten by given `Options` or keyword arguments.

    :param default_options: Options to be used instead of default values
    :param kwargs:          Keyword arguments to be used instead of default values
    :return:                A dictionary that stores the keyword arguments to be used
    """
    options = default_options if default_options else Options()
    return {
        OPTION_DECIMALS: options.get_int(OPTION_DECIMALS, kwargs.get(OPTION_DECIMALS, 0)),
        OPTION_PERCENTAGE: options.get_bool(OPTION_PERCENTAGE, kwargs.get(OPTION_PERCENTAGE, False)),
        OPTION_THOUSANDS_SEPARATORS: options.get_bool(
            OPTION_THOUSANDS_SEPARATORS, kwargs.get(OPTION_THOUSANDS_SEPARATORS, False)
        ),
    }


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
    return f'{format_value(float(fraction), percentage=True, decimals=decimals)} %'


def format_progress(num_completed: int, num_total: int) -> str:
    """
    Creates and returns a textual representation of a progress consisting of a certain number of steps of which some are
    already completed.

    :param num_completed:   The number of already completed steps
    :param num_total:       The total number of steps
    :return:                The textual representation that has been created
    """
    return f'{num_completed} / {num_total}'
