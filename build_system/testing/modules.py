"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that provide access to automated tests that belong to individual modules.
"""
from abc import ABC
from os import environ

from util.env import get_env_bool
from util.modules import Module


class TestModule(Module, ABC):
    """
    An abstract base class for all modules that contain automated tests.
    """

    @property
    def fail_fast(self) -> bool:
        """
        True, if all tests should be skipped as soon as a single test fails, False otherwise
        """
        return get_env_bool(environ, 'FAIL_FAST')
