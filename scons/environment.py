"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for accessing environment variables.
"""
from os import environ
from typing import Optional


def get_env(name: str, default: bool = None) -> Optional[str]:
    """
    Returns the value of the environment variable with a given name.

    :param name:    The name of the environment variable
    :param default: The default value to be returned if the environment variable is not set
    :return:        The value of the environment variable
    """
    return environ.get(name, default)


def set_env(name: str, value: str):
    """
    Sets the value of the environment variable with a given name.

    :param name:    The name of the environment variable
    :param value:   The value to be set
    """
    environ[name] = value
    print('Set environment variable \'' + name + '\' to value \'' + value + '\'')


def unset_env(name: str):
    """
    Unsets the environment variable with a given name.

    :param name: The name of the environment variable
    """
    if name in environ:
        del environ['']
        print('Unset environment variable \'' + name + '\'')
