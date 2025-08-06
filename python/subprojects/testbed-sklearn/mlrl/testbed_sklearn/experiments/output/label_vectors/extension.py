"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write label vectors to one or several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import Set, override

from mlrl.testbed_sklearn.experiments.output.label_vectors.label_vectors import LabelVectors

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument


class LabelVectorExtension(Extension):
    """
    An extension that configures the functionality to write label vectors to one or several sinks.
    """

    PRINT_LABEL_VECTORS = BoolArgument(
        '--print-label-vectors',
        description='Whether the unique label vectors contained in the training data should be printed on the console '
        + 'or not.',
        true_options={LabelVectors.OPTION_SPARSE},
    )

    SAVE_LABEL_VECTORS = BoolArgument(
        '--save-label-vectors',
        description='Whether the unique label vectors contained in the training data should be written to output files '
        + 'or not.',
        true_options={LabelVectors.OPTION_SPARSE},
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_LABEL_VECTORS, self.SAVE_LABEL_VECTORS}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_label_vectors, options = self.PRINT_LABEL_VECTORS.get_value(args, default=print_all)

        if print_label_vectors:
            experiment_builder.label_vector_writer.add_sinks(LogSink(options))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        save_all = OutputArguments.SAVE_ALL.get_value(args)
        save_label_vectors, options = self.SAVE_LABEL_VECTORS.get_value(args, default=save_all)
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if save_label_vectors and result_directory:
            create_directory = OutputArguments.CREATE_DIRS.get_value(args)
            experiment_builder.label_vector_writer.add_sinks(
                CsvFileSink(directory=Path(result_directory), create_directory=create_directory, options=options))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_csv_file_sink(args, experiment_builder)
