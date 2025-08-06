"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Defines command line arguments for configuring the functionality to write output data to one or several sinks.
"""
from datetime import datetime
from pathlib import Path

from mlrl.util.cli import BoolArgument, StringArgument


class OutputArguments:
    """
    Defines command line arguments for configuring the functionality to write output data to one or several sinks.
    """

    BASE_DIR = StringArgument(
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

    EXIT_ON_ERROR = BoolArgument(
        '--exit-on-error',
        default=False,
        description='Whether the program should exit if an error occurs while writing experimental results or not.',
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

    RESULT_DIR = StringArgument(
        '--result-dir',
        default='results',
        description='The path to the directory where experimental results should be saved.',
        decorator=lambda args, value: Path(OutputArguments.BASE_DIR.get_value(args)) / value,
    )

    WIPE_RESULT_DIR = BoolArgument(
        '--wipe-result-dir',
        default=True,
        description='Whether all files in the directory specified via the argument ' + RESULT_DIR.name + ' should be '
        + 'deleted before an experiment starts or not.',
    )
