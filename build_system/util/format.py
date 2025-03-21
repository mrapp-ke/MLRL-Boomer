"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
from functools import reduce
from typing import Any, Callable, Iterable


def format_iterable(objects: Iterable[Any],
                    separator: str = ', ',
                    delimiter: str = '',
                    mapping: Callable[[Any], Any] = lambda x: x) -> str:
    """
    Creates and returns a textual representation of objects in an iterable.

    :param objects:     The iterable of objects to be formatted
    :param separator:   The string that should be used as a separator
    :param delimiter:   The string that should be added at the beginning and end of each object
    :param mapping:     An optional function that maps each object in the iterable to another one
    :return:            The textual representation that has been created
    """
    return reduce(lambda aggr, obj: aggr + (separator
                                            if aggr else '') + delimiter + str(mapping(obj)) + delimiter, objects, '')
