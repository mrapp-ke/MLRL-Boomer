"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
from functools import reduce
from typing import Dict, Iterable, Set


def format_string_iterable(strings: Iterable[str], separator: str = ', ', delimiter: str = '') -> str:
    """
    Creates and returns a textual representation of string in an iterable.

    :params strings:    The iterable of strings to be formatted
    :param separator:   The string that should be used as a separator
    :param delimiter:   The string that should be added at the beginning and end of each string
    :return:            The textual representation that has been created
    """
    return reduce(lambda a, b: a + (separator if len(a) > 0 else '') + delimiter + b + delimiter, strings, '')


def format_enum_values(enum) -> str:
    """
    Creates and returns a textual representation of an enum's values.

    :param enum:    The enum to be formatted
    :return:        The textual representation that has been created
    """
    return format_string_set({x.value for x in enum})


def format_string_set(strings: Set[str]) -> str:
    """
    Creates and returns a textual representation of the strings in a set.

    :param strings: The set of strings to be formatted
    :return:        The textual representation that has been created
    """
    return '{' + format_string_iterable(strings, delimiter='"') + '}'


def format_dict_keys(dictionary: Dict[str, Set[str]]) -> str:
    """
    Creates and returns a textual representation of the keys in a dictionary.

    :param dictionary:  The dictionary to be formatted
    :return:            The textual representation that has been created
    """
    return format_string_set(dictionary.keys())
