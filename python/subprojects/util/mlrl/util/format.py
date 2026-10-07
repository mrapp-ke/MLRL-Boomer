"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""

from collections.abc import Iterable
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np


def format_iterable(objects: Iterable[Any], separator: str = ', ', delimiter: str = '') -> str:
    """
    Creates and returns a textual representation of objects in an iterable.

    :param objects:     The iterable of objects to be formatted
    :param separator:   The string that should be used as a separator
    :param delimiter:   The string that should be added at the beginning and end of each object
    :return:            The textual representation that has been created
    """
    return reduce(lambda aggr, obj: aggr + (separator if aggr else '') + delimiter + str(obj) + delimiter, objects, '')


def format_list(
    objects: list[Any], separator: str = ',  ', delimiter: str = '', last_separator: str | None = None
) -> str:
    """
    Creates and returns a textual representation of objects in a list.

    :param objects:         The list of objects to be formatted
    :param separator:       The string that should be used as a separator
    :param delimiter:       The string that should be added at the beginning and end of each object
    :param last_separator:  The string that should be used as the last separator or None, if `separator` should be used
    :return:                The textual representation that has been created
    """
    last_separator = separator if last_separator is None else last_separator
    return reduce(
        lambda aggr, entry: (
            aggr
            + ((last_separator if entry[0] == len(objects) - 1 else separator) if aggr else '')
            + delimiter
            + str(entry[1])
            + delimiter
        ),
        enumerate(objects),
        '',
    )


def format_enum_values(enum: type[Enum]) -> str:
    """
    Creates and returns a textual representation of an enum's values.

    :param enum:    The enum to be formatted
    :return:        The textual representation that has been created
    """
    values = {x.value if isinstance(x.value, str) else x.name.lower() for x in enum}
    return format_set(values)


def format_set(objects: Iterable[Any]) -> str:
    """
    Creates and returns a textual representation of the objects in a set.

    :param objects: The iterable of objects to be formatted
    :return:        The textual representation that has been created
    """
    return f'{{{format_iterable(sorted(objects, key=str), delimiter='"')}}}'


def format_value(value: Any, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of a given value.

    :param value:       The value to be formatted
    :param decimals:    The number of decimals floating point values should be rounded to or 0, if the number of
                        decimals should not be restricted
    :return:            The textual representation that has been created
    """
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, Path):
        return f'"{value}"'
    if isinstance(value, (float, np.floating)) and decimals > 0:
        rounded_value = round(value, decimals)
        return f'{rounded_value:.{decimals}f}'
    return str(value)
