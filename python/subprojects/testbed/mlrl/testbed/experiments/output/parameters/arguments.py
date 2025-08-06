"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the directory to which algorithmic parameters should be written.
"""
from pathlib import Path

from mlrl.testbed.experiments.output.arguments import OutputArguments

from mlrl.util.cli import StringArgument


class ParameterOutputDirectoryArguments:
    """
    Defines command line arguments for configuring the directory to which algorithmic parameters should be written.
    """

    PARAMETER_SAVE_DIR = StringArgument(
        '--parameter-save-dir',
        default='parameters',
        description='The path to the directory where configuration files, which specify the parameters used by the '
        + 'algorithm, should be saved.',
        decorator=lambda args, value: Path(OutputArguments.BASE_DIR.get_value(args)) / value,
    )
