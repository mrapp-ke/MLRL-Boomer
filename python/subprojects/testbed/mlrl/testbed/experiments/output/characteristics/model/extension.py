"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of rule models to one or several
sinks.
"""
from argparse import ArgumentParser, Namespace
from typing import List

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.characteristics.model.extractor_rules import RuleModelCharacteristicsExtractor
from mlrl.testbed.experiments.output.characteristics.model.writer import ModelCharacteristicsWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.format import format_enum_values
from mlrl.util.options import BooleanOption


class RuleModelCharacteristicsExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of rule models to one or several sinks.
    """

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(
            '--print-model-characteristics',
            type=BooleanOption.parse,
            default=BooleanOption.FALSE.value,
            help='Whether the characteristics of models should be printed on the console or not. Must be one of '
            + format_enum_values(BooleanOption) + '.')
        argument_parser.add_argument(
            '--store-model-characteristics',
            type=BooleanOption.parse,
            default=BooleanOption.FALSE.value,
            help='Whether the characteristics of models should be written into output files or not. Must be one of '
            + format_enum_values(BooleanOption) + '. Does only have an effect if the argument '
            + OutputExtension.PARAM_OUTPUT_DIR + ' is specified.')

    @staticmethod
    def __create_log_sinks(args: Namespace) -> List[Sink]:
        if args.print_model_characteristics or args.print_all:
            return [LogSink()]
        return []

    @staticmethod
    def __create_csv_file_sinks(args: Namespace) -> List[Sink]:
        if (args.store_model_characteristics or args.store_all) and args.output_dir:
            return [CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir)]
        return []

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = ModelCharacteristicsWriter(RuleModelCharacteristicsExtractor(),
                                                exit_on_error=args.exit_on_error).add_sinks(*sinks)
            experiment.add_post_training_output_writers(writer)
