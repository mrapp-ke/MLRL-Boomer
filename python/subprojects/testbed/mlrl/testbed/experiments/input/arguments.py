"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to read input data from one or several sources.
"""
from mlrl.testbed.experiments.input.policies import MissingInputPolicy

from mlrl.util.cli import EnumArgument


class InputArguments:
    """
    Defines command line arguments for configuring the functionality to read input data from one or several sources.
    """

    IF_INPUT_MISSING = EnumArgument(
        '--if-input-missing',
        enum=MissingInputPolicy,
        default=MissingInputPolicy.LOG,
        description='What to do if an error occurs while reading input data.',
    )
