"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import getenv
from unittest import SkipTest


def only_on_ci(decorated_function):
    """
    A decorator that disables all annotated test case unless run on a continuous integration system.
    """

    def wrapper(*args, **kwargs):
        if getenv('GITHUB_ACTIONS') != 'true':
            raise SkipTest('Test is disabled unless run on a CI system')

        decorated_function(*args, **kwargs)

    return wrapper
