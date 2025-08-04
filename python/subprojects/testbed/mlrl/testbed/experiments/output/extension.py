"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write output data to one or several sinks.
"""
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import Mode, SingleMode

from mlrl.util.cli import Argument, BoolArgument, StringArgument


class OutputExtension(Extension):
    """
    An extension that configures the functionality to write output data to one or several sinks.
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

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.BASE_DIR, self.CREATE_DIRS, self.EXIT_ON_ERROR, self.PRINT_ALL, self.SAVE_ALL}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.set_exit_on_error(self.EXIT_ON_ERROR.get_value(args))


class ResultDirectoryExtension(Extension):
    """
    An extension that configures the directory to which experimental results should be written.
    """

    class WipeDirectoryListener(Experiment.Listener):
        """
        Deletes all files from a directory before an experiment starts.
        """

        def __init__(self, directory: Path):
            """
            :param directory: The path to the directory from which the files should be deleted
            """
            self.directory = directory

        @override
        def before_start(self, _: Experiment):
            """
            See :func:`mlrl.testbed.experiments.Experiment.Listener.before_start`
            """
            directory = self.directory

            if directory.is_dir():
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_path.unlink()

    RESULT_DIR = StringArgument(
        '--result-dir',
        default='results',
        description='The path to the directory where experimental results should be saved.',
        decorator=lambda args, value: Path(OutputExtension.BASE_DIR.get_value(args)) / value,
    )

    WIPE_RESULT_DIR = BoolArgument(
        '--wipe-result-dir',
        default=True,
        description='Whether all files in the directory specified via the argument ' + RESULT_DIR.name + ' should be '
        + 'deleted before an experiment starts or not.',
    )

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.RESULT_DIR, self.WIPE_RESULT_DIR}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        result_directory = self.RESULT_DIR.get_value(args)

        if result_directory and self.WIPE_RESULT_DIR.get_value(args):
            listener = ResultDirectoryExtension.WipeDirectoryListener(Path(result_directory))
            experiment_builder.add_listeners(listener)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode}
