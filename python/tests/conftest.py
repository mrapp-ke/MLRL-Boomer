"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""

ARGUMENT_NUM_BLOCKS = '--num-blocks'


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
    for i, item in enumerate(items):
        config = item.config
        num_blocks = config.getoption(ARGUMENT_NUM_BLOCKS)

        if num_blocks < 1 or num_blocks > 8:
            raise ValueError(f'Argument {ARGUMENT_NUM_BLOCKS} must be at least 1 and at most 8')

        num_tests = len(items)
        num_tests_per_block = num_tests // num_blocks
        current_block_index = i // num_tests_per_block
        item.add_marker(f'block-{current_block_index if current_block_index < num_blocks else 0}')
