"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write algorithmic parameters to one or several sinks.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.cli import Argument
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.parameters.writer import ParameterWriter
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension


class ParameterOutputExtension(Extension):
    """
    An extension that configures the functionality to write algorithmic parameters to one or several sinks.
    """

    PARAM_PARAMETER_SAVE_DIR = '--parameter-save-dir'

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            Argument.string(
                self.PARAM_PARAMETER_SAVE_DIR,
                help='The path to the directory where configuration files, which specify the parameters used by the '
                + 'algorithm, should be saved.',
            ),
            Argument.bool(
                '--print-parameters',
                default=False,
                help='Whether the parameter setting should be printed on the console or not.',
            ),
        ]

    @staticmethod
    def __create_log_sinks(args: Namespace) -> List[Sink]:
        if args.print_parameters or args.print_all:
            return [LogSink()]
        return []

    @staticmethod
    def __create_csv_file_sinks(args: Namespace) -> List[Sink]:
        if args.parameter_save_dir:
            return [CsvFileSink(directory=args.parameter_save_dir, create_directory=args.create_output_dir)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = ParameterWriter().add_sinks(*sinks)
            experiment_builder.add_pre_training_output_writers(writer)
