"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write output data to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment, ExperimentListener
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.state import ExperimentMode, ExperimentState
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument


class OutputExtension(Extension):
    """
    An extension that configures the functionality to write output data to one or several sinks.
    """

    @override
    def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {
            OutputArguments.BASE_DIR, OutputArguments.CREATE_DIRS, OutputArguments.EXIT_ON_ERROR,
            OutputArguments.PRINT_ALL, OutputArguments.SAVE_ALL
        }

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: ExperimentMode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.set_exit_on_error(OutputArguments.EXIT_ON_ERROR.get_value(args))

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.SINGLE, ExperimentMode.BATCH, ExperimentMode.READ, ExperimentMode.RUN}


class ResultDirectoryExtension(Extension):
    """
    An extension that configures the directory to which experimental results should be written.
    """

    class WipeDirectoryListener(ExperimentListener):
        """
        Deletes all files from a directory before an experiment starts.
        """

        def __init__(self, directory: Path):
            """
            :param directory: The path to the directory from which the files should be deleted
            """
            self.directory = directory

        @override
        def before_start(self, state: ExperimentState):
            """
            See :func:`mlrl.testbed.experiments.ExperimentListener.before_start`
            """
            directory = self.directory

            if directory.is_dir():
                for file_path in directory.iterdir():
                    if file_path.is_file():
                        file_path.unlink()

            return state

    @override
    def _get_arguments(self, mode: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {ResultDirectoryArguments.RESULT_DIR} if mode == ExperimentMode.READ else {
            ResultDirectoryArguments.RESULT_DIR, ResultDirectoryArguments.WIPE_RESULT_DIR
        }

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, mode: ExperimentMode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        if mode != ExperimentMode.READ:
            base_dir = OutputArguments.BASE_DIR.get_value(args)
            result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

            if base_dir and result_directory and ResultDirectoryArguments.WIPE_RESULT_DIR.get_value(args):
                listener = ResultDirectoryExtension.WipeDirectoryListener(base_dir / result_directory)
                experiment_builder.add_listeners(listener)

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.SINGLE, ExperimentMode.READ, ExperimentMode.RUN}
