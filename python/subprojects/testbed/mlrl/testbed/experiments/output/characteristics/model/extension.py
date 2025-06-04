"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of rule models to one or several
sinks.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.characteristics.model.extractor_rules import RuleModelCharacteristicsExtractor
from mlrl.testbed.experiments.output.characteristics.model.writer import ModelCharacteristicsWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument


class RuleModelCharacteristicsExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of rule models to one or several sinks.
    """

    PRINT_MODEL_CHARACTERISTICS = BoolArgument(
        '--print-model-characteristics',
        default=False,
        description='Whether the characteristics of models should be printed on the console or not.',
    )

    STORE_MODEL_CHARACTERISTICS = BoolArgument(
        '--store-model-characteristics',
        default=False,
        description='Whether the characteristics of models should be written into output files or not. Does only have '
        + 'an effect if the argument ' + OutputExtension.OUTPUT_DIR.name + ' is specified.',
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
        return [self.PRINT_MODEL_CHARACTERISTICS, self.STORE_MODEL_CHARACTERISTICS]

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        if self.PRINT_MODEL_CHARACTERISTICS.get_value(args) or args.print_all:
            return [LogSink()]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        store_model_characteristics = self.STORE_MODEL_CHARACTERISTICS.get_value(args)
        output_dir = OutputExtension.OUTPUT_DIR.get_value(args)

        if (store_model_characteristics or args.store_all) and output_dir:
            return [CsvFileSink(directory=output_dir, create_directory=args.create_output_dir)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = ModelCharacteristicsWriter(RuleModelCharacteristicsExtractor()).add_sinks(*sinks)
            experiment_builder.add_post_training_output_writers(writer)
