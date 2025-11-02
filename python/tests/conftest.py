"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from sys import platform

import numpy as np
import pytest

ARGUMENT_NUM_BLOCKS = '--num-blocks'

MAX_BLOCKS = 32


def pytest_addoption(parser):
    """
    Configures custom arguments to be passed to "pytest".

    :param parser: The argument parser that allows configuring the arguments
    """
    parser.addoption(ARGUMENT_NUM_BLOCKS, type=int, default=1, help='Total number of blocks to divide tests into.')


def pytest_collection_modifyitems(items):
    """
    Assigns test cases to blocks and assigns them a corresponding marker.

    :param items: The test cases
    """
    if not platform.startswith('linux'):
        pytest.skip('Integration tests are only supported on Linux', allow_module_level=True)

    num_tests = len(items)
    block_indices = None

    for i, item in enumerate(items):
        config = item.config
        num_blocks = config.getoption(ARGUMENT_NUM_BLOCKS)

        if block_indices is None:
            seed = 42
            rng = np.random.default_rng(seed)
            block_indices = rng.integers(low=0, high=num_blocks, size=num_tests)

        if num_blocks < 1 or num_blocks > MAX_BLOCKS:
            raise ValueError(f'Argument {ARGUMENT_NUM_BLOCKS} must be at least 1 and at most {MAX_BLOCKS}')

        item.add_marker(f'block-{block_indices[i]}')
