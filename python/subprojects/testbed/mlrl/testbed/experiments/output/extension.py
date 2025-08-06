"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write output data to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.state import ExperimentState
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import Mode, SingleMode

from mlrl.util.cli import Argument


class OutputExtension(Extension):
    """
    An extension that configures the functionality to write output data to one or several sinks.
    """

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {
            OutputArguments.BASE_DIR, OutputArguments.CREATE_DIRS, OutputArguments.EXIT_ON_ERROR,
            OutputArguments.PRINT_ALL, OutputArguments.SAVE_ALL
        }

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.set_exit_on_error(OutputArguments.EXIT_ON_ERROR.get_value(args))


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
        def before_start(self, _: Experiment, state: ExperimentState):
            """
            See :func:`mlrl.testbed.experiments.Experiment.Listener.before_start`
            """
            directory = self.directory

            if directory.is_dir():
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_path.unlink()

            return state

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {ResultDirectoryArguments.RESULT_DIR, ResultDirectoryArguments.WIPE_RESULT_DIR}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if result_directory and ResultDirectoryArguments.WIPE_RESULT_DIR.get_value(args):
            listener = ResultDirectoryExtension.WipeDirectoryListener(Path(result_directory))
            experiment_builder.add_listeners(listener)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode}
