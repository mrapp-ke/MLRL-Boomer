"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with dates and times.
"""

from datetime import UTC, datetime, timezone


def get_default_timezone() -> timezone:
    """
    Returns the timezone used by default.

    :return: The timezone used by default
    """
    return UTC


def get_current_datetime() -> datetime:
    """
    Returns the current date and time using the default timezone.

    :return: The current date and time
    """
    return datetime.now(get_default_timezone())
