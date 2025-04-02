"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
import sys

from functools import reduce
from typing import List, Optional

import numpy as np

from tabulate import tabulate

OPTION_DECIMALS = 'decimals'

OPTION_PERCENTAGE = 'percentage'


def format_duration(duration: float) -> str:
    """
    Creates and returns a textual representation of a duration.

    :param duration:    The duration in seconds
    :return:            The textual representation that has been created
    """
    seconds, millis = divmod(duration, 1)
    millis = int(millis * 1000)
    days, seconds = divmod(seconds, 86400)
    days = int(days)
    hours, seconds = divmod(seconds, 3600)
    hours = int(hours)
    minutes, seconds = divmod(seconds, 60)
    minutes = int(minutes)
    seconds = int(seconds)
    substrings = []

    if days > 0:
        substrings.append(str(days) + ' day' + ('' if days == 1 else 's'))

    if hours > 0:
        substrings.append(str(hours) + ' hour' + ('' if hours == 1 else 's'))

    if minutes > 0:
        substrings.append(str(minutes) + ' minute' + ('' if minutes == 1 else 's'))

    if seconds > 0:
        substrings.append(str(seconds) + ' second' + ('' if seconds == 1 else 's'))

    if millis > 0 or len(substrings) == 0:
        substrings.append(str(millis) + ' millisecond' + ('' if millis == 1 else 's'))

    return reduce(lambda aggr, x: aggr + ((' and ' if x[0] == len(substrings) - 1 else ', ') if aggr else '') + x[1],
                  enumerate(substrings), '')


def format_array(array: np.ndarray, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of an array.

    :param array:       The array
    :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
    :return:            The textual representation that has been created
    """
    if array.dtype.kind == 'f':
        precision = decimals if decimals > 0 else None
        return np.array2string(array, threshold=sys.maxsize, precision=precision, suppress_small=True)
    # pylint: disable=unnecessary-lambda
    return np.array2string(array, threshold=sys.maxsize, formatter={'all': lambda x: str(x)})


def format_float(value: float, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of a floating point value using a specific number of decimals.

    :param value:       The value
    :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
    :return:            The textual representation that has been created
    """
    return ('{:.' + str(decimals) + 'f}').format(round(value, decimals)) if decimals > 0 else str(value)


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Creates and returns a textual representation of a percentage using a specific number of decimals.

    :param value:       The percentage
    :param decimals:    The number of decimals to be used or 0, if the number of decimals should not be restricted
    :return:            The textual representation that has been created
    """
    return format_float(value, decimals) + '%'


def format_table(rows: List[List[str]], header: Optional[List[str]] = None, alignment: List[str] = None) -> str:
    """
    Creates and returns a textual representation of tabular data.

    :param rows:        A list of lists that stores the tabular data
    :param header:      A list that stores the header columns
    :param alignment:   A list of strings that specify the alignment of the corresponding column as either 'left',
                        'center', or 'right'
    :return:            The textual representation that has been created
    """
    if not header:
        return tabulate(rows, colalign=alignment, tablefmt='plain')
    return tabulate(rows, headers=header, colalign=alignment, tablefmt='simple_outline')
