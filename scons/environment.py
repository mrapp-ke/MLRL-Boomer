"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for accessing environment variables.
"""
from typing import List, Optional


def get_env(env, name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Returns the value of the environment variable with a given name.

    :param env:     The environment to be accessed
    :param name:    The name of the environment variable
    :param default: The default value to be returned if the environment variable is not set
    :return:        The value of the environment variable
    """
    return env.get(name, default)


def get_env_array(env, name: str, default: Optional[List[str]] = None) -> List[str]:
    """
    Returns the value of the environment variable with a given name as a comma-separated list.

    :param env:     The environment to be accessed
    :param name:    The name of the environment variable
    :param default: The default value to be returned if the environment variable is not set
    :return:        The value of the environment variable
    """
    value = get_env(env, name)

    if value:
        return value.split(',')

    return default if default else []


def set_env(env, name: str, value: str):
    """
    Sets the value of the environment variable with a given name.

    :param env:     The environment to be modified
    :param name:    The name of the environment variable
    :param value:   The value to be set
    """
    env[name] = value
    print('Set environment variable \'' + name + '\' to value \'' + value + '\'')
