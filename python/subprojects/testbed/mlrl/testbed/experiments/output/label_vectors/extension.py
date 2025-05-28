"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write label vectors to one or several sinks.
"""
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.label_vectors.extractor_rules import LabelVectorSetExtractor
from mlrl.testbed.experiments.output.label_vectors.label_vectors import LabelVectors
from mlrl.testbed.experiments.output.label_vectors.writer import LabelVectorWriter
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.format import format_set
from mlrl.util.options import BooleanOption, parse_param_and_options


class LabelVectorExtension(Extension):
    """
    An extension that configures the functionality to write label vectors to one or several sinks.
    """

    PARAM_PRINT_LABEL_VECTORS = '--print-label-vectors'

    PRINT_LABEL_VECTORS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {LabelVectors.OPTION_SPARSE},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_LABEL_VECTORS = '--store-label-vectors'

    STORE_LABEL_VECTORS_VALUES = PRINT_LABEL_VECTORS_VALUES

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(
            self.PARAM_PRINT_LABEL_VECTORS,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the unique label vectors contained in the training data should be printed on the console or '
            + 'not. Must be one of ' + format_set(self.PRINT_LABEL_VECTORS_VALUES.keys()) + '. For additional options '
            + 'refer to the documentation.')
        argument_parser.add_argument(
            self.PARAM_STORE_LABEL_VECTORS,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the unique label vectors contained in the training data should be written into output files '
            + 'or not. Must be one of ' + format_set(self.STORE_LABEL_VECTORS_VALUES.keys()) + '. Does only have an '
            + 'effect if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is specified. For additional options '
            + ' refer to the documentation.')

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_LABEL_VECTORS, args.print_label_vectors,
                                                 self.PRINT_LABEL_VECTORS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_LABEL_VECTORS, args.store_label_vectors,
                                                 self.STORE_LABEL_VECTORS_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            return [CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = LabelVectorWriter(LabelVectorSetExtractor()).add_sinks(*sinks)
            writer.exit_on_error = args.exit_on_error
            experiment_builder.add_post_training_output_writers(writer)
