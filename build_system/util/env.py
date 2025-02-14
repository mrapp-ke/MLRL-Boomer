"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides utility functions for accessing environment variables.
"""
from typing import Dict, List, Optional

from util.log import Log


def get_env(env: Dict, name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Returns the value of the environment variable with a given name.

    :param env:     The environment to be accessed
    :param name:    The name of the environment variable
    :param default: The default value to be returned if the environment variable is not set
    :return:        The value of the environment variable
    """
    return env.get(name, default)


def get_env_bool(env: Dict, name: str, default: bool = False) -> bool:
    """
    Returns the value of the environment variable with a given name as a boolean value.

    :param env:     The environment to be accessed
    :param name:    The name of the environment variable
    :param default: The default value to be returned if the environment variable is not set
    :return:        The value of the environment variable
    """
    value = get_env(env, name)

    if value:
        value_lower = value.strip().lower()

        if value_lower == 'true':
            return True
        if value_lower == 'false':
            return False
        raise ValueError('Value of environment variable "' + name + '" must be "true" or "false", but is "' + value
                         + '"')

    return default


def get_env_array(env: Dict, name: str, default: Optional[List[str]] = None) -> List[str]:
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


def set_env(env: Dict, name: str, value: str):
    """
    Sets the value of the environment variable with a given name.

    :param env:     The environment to be modified
    :param name:    The name of the environment variable
    :param value:   The value to be set
    """
    env[name] = value
    Log.info('Set environment variable "%s" to value "%s"', name, value)
