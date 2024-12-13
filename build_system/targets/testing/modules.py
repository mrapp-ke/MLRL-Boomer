"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to automated tests.
"""
from abc import ABC
from os import environ

from core.modules import Module
from util.env import get_env_bool


class TestModule(Module, ABC):
    """
    An abstract base class for all modules that provide access to automated tests.
    """

    @property
    def fail_fast(self) -> bool:
        """
        True, if all tests should be skipped as soon as a single test fails, False otherwise
        """
        return get_env_bool(environ, 'FAIL_FAST')
