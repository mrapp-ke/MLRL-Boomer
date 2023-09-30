"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for dealing with environment variables.
"""
import os

from typing import Optional


def get_bool_env(name: str) -> Optional[bool]:
    """
    Returns a boolean value from an environment variable. If the value of the environment variable is neither "true",
    nor "false", a `ValueError` is raised.

    :param name:    The name of the environment variable
    :return:        True, if the environment variable is set to "true", False if it set to "false", or None if the
                    environment variable is not set
    """
    value = os.environ.get(name, None)

    if not value:
        return None
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise ValueError('Invalid boolean value given for environment variable "' + name + '". Must be "true" or "false".')
