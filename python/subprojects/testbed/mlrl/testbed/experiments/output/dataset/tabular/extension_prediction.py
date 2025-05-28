"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write predictions to one or several sinks.
"""
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Set

from mlrl.testbed_arff.experiments.output.sinks.sink_arff import ArffFileSink

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.dataset.tabular.writer_prediction import PredictionWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.format import format_set
from mlrl.util.options import BooleanOption, parse_param_and_options


class PredictionExtension(Extension):
    """
    An extension that configures the functionality to write predictions to one or several sinks.
    """

    PARAM_PRINT_PREDICTIONS = '--print-predictions'

    PRINT_PREDICTIONS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_PREDICTIONS = '--store-predictions'

    STORE_PREDICTIONS_VALUES = PRINT_PREDICTIONS_VALUES

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(self.PARAM_PRINT_PREDICTIONS,
                                     type=str,
                                     default=BooleanOption.FALSE.value,
                                     help='Whether predictions should be printed on the console or not. Must be one of '
                                     + format_set(self.PRINT_PREDICTIONS_VALUES.keys())
                                     + '. For additional options refer to the documentation.')
        argument_parser.add_argument(
            self.PARAM_STORE_PREDICTIONS,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether predictions should be written into output files or not. Must be one of '
            + format_set(self.STORE_PREDICTIONS_VALUES.keys()) + '. Does only have an effect, if the argument '
            + OutputExtension.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the documentation.')

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_PREDICTIONS, args.print_predictions,
                                                 self.PRINT_PREDICTIONS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options=options)]
        return []

    def __create_arff_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_PREDICTIONS, args.store_predictions,
                                                 self.STORE_PREDICTIONS_VALUES)

        if value == BooleanOption.TRUE.value and args.output_dir:
            return [ArffFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_arff_file_sinks(args)

        if sinks:
            writer = PredictionWriter().add_sinks(*sinks)
            experiment.add_prediction_output_writers(writer)
