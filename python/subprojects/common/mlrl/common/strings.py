"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with strings.
"""
from functools import reduce
from typing import Set, Dict


def format_enum_values(enum) -> str:
    """
    Creates and returns a textual representation of an enum's values.

    :param enum:    The enum to be formatted
    :return:        The textual representation that has been created
    """
    return '{' + reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b.value + '"', enum, '') + '}'


def format_string_set(strings: Set[str]) -> str:
    """
    Creates and returns a textual representation of the strings in a set.

    :param strings: The set of strings to be formatted
    :return:        The textual representation that has been created
    """
    return '{' + reduce(lambda a, b: a + (', ' if len(a) > 0 else '') + '"' + b + '"', strings, '') + '}'


def format_dict_keys(dictionary: Dict[str, Set[str]]) -> str:
    """
    Creates and returns a textual representation of the keys in a dictionary.

    :param dictionary:  The dictionary to be formatted
    :return:            The textual representation that has been created
    """
    return format_string_set(set(dictionary.keys()))


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

    return reduce(
        lambda txt, x: txt + ((' and ' if x[0] == len(substrings) - 1 else ', ') if len(txt) > 0 else '') + x[1],
        enumerate(substrings), '')
