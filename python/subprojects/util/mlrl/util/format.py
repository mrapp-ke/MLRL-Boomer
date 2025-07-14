"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
from enum import Enum
from functools import reduce
from typing import Any, Iterable, Type


def format_iterable(objects: Iterable[Any], separator: str = ', ', delimiter: str = '') -> str:
    """
    Creates and returns a textual representation of objects in an iterable.

    :param objects:     The iterable of objects to be formatted
    :param separator:   The string that should be used as a separator
    :param delimiter:   The string that should be added at the beginning and end of each object
    :return:            The textual representation that has been created
    """
    return reduce(lambda aggr, obj: aggr + (separator if aggr else '') + delimiter + str(obj) + delimiter, objects, '')


def format_enum_values(enum: Type[Enum]) -> str:
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
    return '{' + format_iterable(sorted(objects, key=str), delimiter='"') + '}'
