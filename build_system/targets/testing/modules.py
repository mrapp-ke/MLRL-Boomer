"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Implements modules that provide access to automated tests.
"""
from abc import ABC
from os import environ
from typing import Optional, Set

from core.modules import Module
from util.env import get_env, get_env_array, get_env_bool


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

    @property
    def only_failed(self) -> bool:
        """
        True, if only tests that failed during the last run should be executed, False otherwise.
        """
        return get_env_bool(environ, 'ONLY_FAILED')

    @property
    def markers(self) -> Set[str]:
        """
        A set that contains the markers of the test cases to be run or an empty list, if all test cases should be run.
        """
        markers = get_env_array(environ, 'MARKERS')
        return set(markers) if markers else set()

    @property
    def num_blocks(self) -> Optional[int]:
        """
        The total number of blocks to assign tests to or None, if no blocks should be used
        """
        value = get_env(environ, 'NUM_BLOCKS')
        return int(value) if value else None

    @property
    def block_index(self) -> Optional[int]:
        """
        The index of the block of tests to be run or None, if all tests should be run.
        """
        value = get_env(environ, 'BLOCK_INDEX')
        return int(value) if value else None
