"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write label vectors to one or several sinks.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vectors import LabelVectors

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks import CsvFileSink, LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode, SingleMode

from mlrl.util.cli import Argument, BoolArgument


class LabelVectorExtension(Extension):
    """
    An extension that configures the functionality to write label vectors to one or several sinks.
    """

    @staticmethod
    def __create_argument_print_label_vectors(mode: Mode) -> BoolArgument:
        return BoolArgument(
            '--print-label-vectors',
            description='Whether the unique label vectors contained in the training data should be printed on the '
            + 'console or not.',
            true_options=None if isinstance(mode, ReadMode) else {LabelVectors.OPTION_SPARSE},
        )

    @staticmethod
    def __create_argument_save_label_vectors(mode: Mode) -> BoolArgument:
        return BoolArgument(
            '--save-label-vectors',
            description='Whether the unique label vectors contained in the training data should be written to output '
            + 'files or not.',
            true_options=None if isinstance(mode, ReadMode) else {LabelVectors.OPTION_SPARSE},
        )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self, mode: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.__create_argument_print_label_vectors(mode), self.__create_argument_save_label_vectors(mode)}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder, mode: Mode):
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_label_vectors, options = self.__create_argument_print_label_vectors(mode).get_value_and_options(
            args, default=print_all)

        if print_label_vectors:
            experiment_builder.label_vector_writer.add_sinks(LogSink(options))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder, mode: Mode):
        save_all = OutputArguments.SAVE_ALL.get_value(args)
        save_label_vectors, options = self.__create_argument_save_label_vectors(mode).get_value_and_options(
            args, default=save_all)
        base_dir = OutputArguments.BASE_DIR.get_value(args)
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if save_label_vectors and base_dir and result_directory:
            create_directory = OutputArguments.CREATE_DIRS.get_value(args)
            experiment_builder.label_vector_writer.add_sinks(
                CsvFileSink(directory=base_dir / result_directory, create_directory=create_directory, options=options))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, mode: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder, mode)
        self.__configure_csv_file_sink(args, experiment_builder, mode)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, ReadMode, RunMode}
