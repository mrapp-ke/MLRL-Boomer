"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with environment variables.
"""
import os

from typing import Optional, Set


def get_string_env(name: str, accepted_values: Optional[Set[str]] = None) -> Optional[str]:
    """
    Returns a string value from an environment variable. If the value of the environment variable is not included in the
    given set of accepted values, a `ValueError` is raised.

    :param name:            The name of the environment variable
    :param accepted_values: An optional set of accepted values
    :return:                The value of the environment variable or None if the environment variable is not set
    """
    value = os.environ.get(name, None)

    if value and accepted_values:
        if not value in accepted_values:
            raise ValueError('Invalid value given for environment variable "' + name + '": ' + str(value))

    return value
