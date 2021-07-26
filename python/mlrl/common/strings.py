#!/usr/bin/python

"""
Author: Michael Rapp (mrapp@ke.tu-darmstadt.de)

Provides utility functions for dealing with strings.
"""
from functools import reduce
from typing import List


def format_enum_values(enum) -> str:
    """
    Creates and returns a textual representation of an enum's values.

    :param enum:    The enum to be formatted
    :return:        The textual representation that has been created
    """
    return '{' + reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b.value + '"', enum, '') + '}'


def format_string_list(strings: List[str]) -> str:
    """
    Creates and returns a textual representation of the strings in a list.

    :param strings: The list of strings to be formatted
    :return:        The textual representation that has been created
    """
    return '{' + reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b + '"', strings, '') + '}'
