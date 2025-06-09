"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write algorithmic parameters to one or several sinks.
"""
from argparse import Namespace
from typing import List, Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.parameters.writer import ParameterWriter
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument, StringArgument


class ParameterOutputExtension(Extension):
    """
    An extension that configures the functionality to write algorithmic parameters to one or several sinks.
    """

    PARAMETER_SAVE_DIR = StringArgument(
        '--parameter-save-dir',
        description='The path to the directory where configuration files, which specify the parameters used by the '
        + 'algorithm, should be saved.',
    )

    PRINT_PARAMETERS = BoolArgument(
        '--print-parameters',
        default=False,
        description='Whether the parameter setting should be printed on the console or not.',
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PARAMETER_SAVE_DIR, self.PRINT_PARAMETERS}

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        if self.PRINT_PARAMETERS.get_value(args, default=args.print_all):
            return [LogSink()]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        parameter_save_dir = self.PARAMETER_SAVE_DIR.get_value(args)

        if parameter_save_dir:
            return [
                CsvFileSink(directory=parameter_save_dir,
                            create_directory=OutputExtension.CREATE_OUTPUT_DIR.get_value(args))
            ]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = ParameterWriter().add_sinks(*sinks)
            experiment_builder.add_pre_training_output_writers(writer)
