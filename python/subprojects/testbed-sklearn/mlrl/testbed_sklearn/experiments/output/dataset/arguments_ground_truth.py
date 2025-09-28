"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to write the ground truth for tabular data to one or
several sinks.
"""
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.cli import BoolArgument


class GroundTruthArguments:
    """
    Defines command line arguments for configuring the functionality to write the ground truth for tabular data to one
    or several sinks.
    """

    PRINT_GROUND_TRUTH = BoolArgument(
        '--print-ground-truth',
        description='Whether the ground truth should be printed on the console or not.',
        true_options={OPTION_DECIMALS},
    )

    SAVE_GROUND_TRUTH = BoolArgument(
        '--save-ground-truth',
        description='Whether the ground truth should be written to output files or not.',
        true_options={OPTION_DECIMALS},
    )
