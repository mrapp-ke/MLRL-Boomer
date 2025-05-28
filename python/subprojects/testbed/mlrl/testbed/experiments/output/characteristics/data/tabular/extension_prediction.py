"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of binary predictions to one or
several sinks.
"""
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.characteristics.data.tabular.characteristics import OutputCharacteristics
from mlrl.testbed.experiments.output.characteristics.data.tabular.writer_prediction import \
    PredictionCharacteristicsWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.format import format_set
from mlrl.util.options import BooleanOption, parse_param_and_options


class PredictionCharacteristicsExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of binary predictions to one or several
    sinks.
    """

    # TODO remove
    PARAM_PREDICTION_TYPE = '--prediction-type'

    PARAM_PRINT_PREDICTION_CHARACTERISTICS = '--print-prediction-characteristics'

    PRINT_PREDICTION_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTION_CHARACTERISTICS = '--store-prediction-characteristics'

    STORE_PREDICTION_CHARACTERISTICS_VALUES = PRINT_PREDICTION_CHARACTERISTICS_VALUES

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(
            self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the characteristics of binary predictions should be printed on the console or not. Must be '
            + 'one of ' + format_set(self.PRINT_PREDICTION_CHARACTERISTICS_VALUES.keys()) + '. Does only have an '
            + 'effect if the argument ' + self.PARAM_PREDICTION_TYPE + ' is set to ' + PredictionType.BINARY.value
            + '. For additional options refer to the documentation.')
        argument_parser.add_argument(
            self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the characteristics of binary predictions should be written into output files or not. Must '
            + 'be one of ' + format_set(self.STORE_PREDICTION_CHARACTERISTICS_VALUES.keys()) + '. Does only have an '
            + 'effect if the argument ' + self.PARAM_PREDICTION_TYPE + ' is set to ' + PredictionType.BINARY.value
            + ' and if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is specified. For additional options '
            + 'refer to the documentation.')

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTION_CHARACTERISTICS,
                                                 args.print_prediction_characteristics,
                                                 self.PRINT_PREDICTION_CHARACTERISTICS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTION_CHARACTERISTICS,
                                                 args.store_prediction_characteristics,
                                                 self.STORE_PREDICTION_CHARACTERISTICS_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            return [CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = PredictionCharacteristicsWriter().add_sinks(*sinks)
            experiment_builder.add_prediction_output_writers(writer)
