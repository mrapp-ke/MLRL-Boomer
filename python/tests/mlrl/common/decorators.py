"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from os import getenv
from unittest import SkipTest


def skip_test_on_ci(decorated_function):
    """
    A decorator that disables all annotated test case if run on a continuous integration system.
    """

    def wrapper(*args, **kwargs):
        if getenv('GITHUB_ACTIONS') == 'true':
            raise SkipTest('Temporarily disabled when run on CI')

        decorated_function(*args, **kwargs)

    return wrapper
