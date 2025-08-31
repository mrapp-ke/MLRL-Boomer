"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to read input data from one or several sources.
"""
from mlrl.util.cli import BoolArgument


class InputArguments:
    """
    Defines command line arguments for configuring the functionality to read input data from one or several sources.
    """

    EXIT_ON_MISSING_INPUT = BoolArgument(
        '--exit-on-missing-input',
        default=False,
        description='Whether the program should exit if an error occurs while reading input data or not.',
    )
