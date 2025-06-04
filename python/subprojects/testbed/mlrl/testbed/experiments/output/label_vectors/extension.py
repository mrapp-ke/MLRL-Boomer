"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write label vectors to one or several sinks.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.label_vectors.extractor_rules import LabelVectorSetExtractor
from mlrl.testbed.experiments.output.label_vectors.label_vectors import LabelVectors
from mlrl.testbed.experiments.output.label_vectors.writer import LabelVectorWriter
from mlrl.testbed.experiments.output.sinks.sink import Sink
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
        default=False,
        description='Whether the unique label vectors contained in the training data should be printed on the console '
        + 'or not.',
        true_options={LabelVectors.OPTION_SPARSE},
    )

    STORE_LABEL_VECTORS = BoolArgument(
        '--store-label-vectors',
        default=False,
        description='Whether the unique label vectors contained in the training data should be written into output '
        + 'files or not. Does only have an effect if the argument ' + OutputExtension.OUTPUT_DIR.name + ' is '
        + 'specified.',
        true_options={LabelVectors.OPTION_SPARSE},
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    def _get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return [self.PRINT_LABEL_VECTORS, self.STORE_LABEL_VECTORS]

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.PRINT_LABEL_VECTORS.get_value(args)

        if value or (value is None and args.print_all):
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.STORE_LABEL_VECTORS.get_value(args)
        output_dir = OutputExtension.OUTPUT_DIR.get_value(args)

        if (value or (value is None and args.store_all)) and output_dir:
            return [CsvFileSink(directory=output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = LabelVectorWriter(LabelVectorSetExtractor()).add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)
