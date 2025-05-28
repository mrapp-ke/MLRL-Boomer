"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of rule models to one or several
sinks.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.cli import Argument
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.characteristics.model.extractor_rules import RuleModelCharacteristicsExtractor
from mlrl.testbed.experiments.output.characteristics.model.writer import ModelCharacteristicsWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension


class RuleModelCharacteristicsExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of rule models to one or several sinks.
    """

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            Argument.bool(
                '--print-model-characteristics',
                default=False,
                help='Whether the characteristics of models should be printed on the console or not.',
            ),
            Argument.bool(
                '--store-model-characteristics',
                default=False,
                help='Whether the characteristics of models should be written into output files or not. Does only have '
                + 'an effect if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is specified.',
            ),
        ]

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

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = ModelCharacteristicsWriter(RuleModelCharacteristicsExtractor()).add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)
