"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the directory to which algorithmic parameters should be written.
"""
from mlrl.util.cli import PathArgument


class ParameterOutputDirectoryArguments:
    """
    Defines command line arguments for configuring the directory to which algorithmic parameters should be written.
    """

    PARAMETER_SAVE_DIR = PathArgument(
        '--parameter-save-dir',
        default='parameters',
        description='The path to the directory where configuration files, which specify the parameters used by the '
        + 'algorithm, should be saved.',
    )
