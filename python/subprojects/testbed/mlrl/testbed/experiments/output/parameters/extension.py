"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write algorithmic parameters to one or several sinks.
"""
from argparse import Namespace
from typing import Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
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
        default='parameters',
        description='The path to the directory where configuration files, which specify the parameters used by the '
        + 'algorithm, should be saved.',
    )

    PRINT_PARAMETERS = BoolArgument(
        '--print-parameters',
        default=False,
        description='Whether the parameter setting should be printed on the console or not.',
    )

    SAVE_PARAMETERS = BoolArgument(
        '--save-parameters',
        default=False,
        description='Whether the parameter setting should be written to output files or not.',
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
        return {self.PARAMETER_SAVE_DIR, self.PRINT_PARAMETERS, self.SAVE_PARAMETERS}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputExtension.PRINT_ALL.get_value(args)
        print_parameters = self.PRINT_PARAMETERS.get_value(args, default=print_all)

        if print_parameters:
            experiment_builder.parameter_writer.add_sinks(LogSink())

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        if self.SAVE_PARAMETERS.get_value(args):
            parameter_save_dir = self.PARAMETER_SAVE_DIR.get_value(args)

            if parameter_save_dir:
                create_directory = OutputExtension.CREATE_DIRS.get_value(args)
                experiment_builder.parameter_writer.add_sinks(
                    CsvFileSink(directory=parameter_save_dir, create_directory=create_directory))

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_csv_file_sink(args, experiment_builder)
