"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for accessing environment variables.
"""
from os import environ
from typing import Optional


def get_env(name: str, default: bool = None, env=environ) -> Optional[str]:
    """
    Returns the value of the environment variable with a given name.

    :param name:    The name of the environment variable
    :param default: The default value to be returned if the environment variable is not set
    :param env:     The environment to be accessed
    :return:        The value of the environment variable
    """
    return env.get(name, default)


def set_env(name: str, value: str, env=environ):
    """
    Sets the value of the environment variable with a given name.

    :param name:    The name of the environment variable
    :param value:   The value to be set
    :param env:     The environment to be modified
    """
    env[name] = value
    print('Set environment variable \'' + name + '\' to value \'' + value + '\'')
