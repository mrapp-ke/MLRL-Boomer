"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to write output data to one or several sinks.
"""
from datetime import datetime
from pathlib import Path

from mlrl.testbed.experiments.output.policies import OutputErrorPolicy, OutputExistsPolicy

from mlrl.util.cli import BoolArgument, EnumArgument, PathArgument


class OutputArguments:
    """
    Defines command line arguments for configuring the functionality to write output data to one or several sinks.
    """

    BASE_DIR = PathArgument(
        '--base-dir',
        default=Path('experiments') / datetime.now().strftime('%Y-%m-%d_%H-%M'),
        description='If relative paths to directories, where files should be saved, are given, they are considered '
        + 'relative to the directory specified via this argument.',
    )

    CREATE_DIRS = BoolArgument(
        '--create-dirs',
        default=True,
        description='Whether the directories, where files should be saved, should be created automatically, if they do '
        + 'not exist, or not.')

    IF_OUTPUT_ERROR = EnumArgument(
        '--if-output-error',
        enum=OutputErrorPolicy,
        default=OutputErrorPolicy.LOG,
        description='Whether the program should exit if an error occurs while writing experimental results or not.',
    )

    IF_OUTPUTS_EXIST = EnumArgument(
        '--if-outputs-exist',
        enum=OutputExistsPolicy,
        default=OutputExistsPolicy.CANCEL,
        description='What to do if experimental results do already exist.',
    )

    PRINT_ALL = BoolArgument(
        '--print-all',
        default=False,
        description='Whether all output data should be printed on the console or not.',
    )

    SAVE_ALL = BoolArgument(
        '--save-all',
        default=False,
        description='Whether all output data should be written to files or not.',
    )


class ResultDirectoryArguments:
    """
    Defines command line arguments for configuring the directory to which experimental results should be written.
    """

    RESULT_DIR = PathArgument(
        '--result-dir',
        default='results',
        description='The path to the directory where experimental results should be saved.',
    )
