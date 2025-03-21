"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
import sys

from functools import reduce
from typing import List

import numpy as np

from tabulate import tabulate

from mlrl.common.config.options import Options

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


def format_table(rows, header=None, alignment=None) -> str:
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


class Formatter:
    """
    Allows to create textual representations of values.
    """

    def __init__(self, option: str, name: str, percentage: bool = False):
        """
        :param option:      The name of the option that can be used for filtering
        :param name:        A name that describes the type of values
        :param percentage:  True, if the values can be formatted as a percentage, False otherwise
        """
        self.option = option
        self.name = name
        self.percentage = percentage

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __hash__(self):
        return hash(self.name)

    def format(self, value, **kwargs) -> str:
        """
        Creates and returns a textual representation of a given value.

        :param value:   The value
        :return:        The textual representation that has been created
        """
        decimals = kwargs.get(OPTION_DECIMALS, 0)

        if self.percentage and kwargs.get(OPTION_PERCENTAGE, False):
            value = value * 100

        return format_float(value, decimals=decimals)


def filter_formatters(formatters: List[Formatter], options_list: List[Options]) -> List[Formatter]:
    """
    Allows to filter a list of `Formatter` objects.

    :param formatters:      A list of `Formatter` objects
    :param options_list:    A list of `Options` objects that should be used for filtering
    :return:                A filtered list of the given `Formatter` objects
    """
    filtered: List[Formatter] = []

    for formatter in formatters:
        if reduce(lambda aggr, options, option=formatter.option: aggr | options.get_bool(option, True),
                  options_list,
                  False):
            filtered.append(formatter)

    return filtered
