"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for creating textual representations.
"""
from functools import reduce


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
