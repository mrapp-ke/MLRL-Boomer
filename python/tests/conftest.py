"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

import inspect

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
    Assigns test cases to blocks using a cost-sensitive round-robin strategy and assigns them a corresponding marker.
    The cost of each test is determined by the number of "mlrl-testbed" invocations in its source code. Tests are
    assigned to the block with the currently lowest total cost, which distributes both test count and expensive tests
    evenly.

    :param items: The test cases
    """
    if not platform.startswith('linux'):
        pytest.skip('Integration tests are only supported on Linux', allow_module_level=True)

    if items:
        num_blocks = items[0].config.getoption(ARGUMENT_NUM_BLOCKS)

        if num_blocks < 1 or num_blocks > MAX_BLOCKS:
            raise ValueError(f'Argument {ARGUMENT_NUM_BLOCKS} must be at least 1 and at most {MAX_BLOCKS}')

        block_costs = np.zeros(num_blocks, dtype=int)

        for item in items:
            cost = __get_test_cost(item)
            block_index = int(np.argmin(block_costs))
            item.add_marker(f'block-{block_index}')
            block_costs[block_index] += cost


def __get_test_cost(item) -> int:
    if 'test_scikit_learn_compatibility' in item.name:
        return 16

    try:
        source = inspect.getsource(item.function)
        cost = 2 if source.find('.set_mode(ExperimentMode.BATCH)') else 1
        multiplier = max(1, source.count('CmdRunner('))
        return multiplier * cost
    except (OSError, TypeError):
        return 1
