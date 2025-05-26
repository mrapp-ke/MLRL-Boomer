"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write ground truth to one or several sinks.
"""
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Set

from mlrl.testbed_arff.experiments.output.sinks.sink_arff import ArffFileSink

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.dataset.tabular.writer_ground_truth import GroundTruthWriter
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.format import format_set
from mlrl.util.options import BooleanOption, parse_param_and_options


class GroundTruthExtension(Extension):
    """
    An extension that configures the functionality to write ground truth to one or several sinks.
    """

    # TODO remove
    PARAM_OUTPUT_DIR = '--output-dir'

    PARAM_PRINT_GROUND_TRUTH = '--print-ground-truth'

    PRINT_GROUND_TRUTH_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_GROUND_TRUTH = '--store-ground-truth'

    STORE_GROUND_TRUTH_VALUES = PRINT_GROUND_TRUTH_VALUES

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(
            self.PARAM_PRINT_GROUND_TRUTH,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the ground truth should be printed on the console or not. Must be one of '
            + format_set(self.PRINT_GROUND_TRUTH_VALUES.keys()) + '. For additional options refer to the '
            + 'documentation.')
        argument_parser.add_argument(
            self.PARAM_STORE_GROUND_TRUTH,
            type=str,
            default=BooleanOption.FALSE.value,
            help='Whether the ground truth should be written into output files or not. Must be one of '
            + format_set(self.STORE_GROUND_TRUTH_VALUES.keys()) + '. Does only have an effect, if the argument '
            + self.PARAM_OUTPUT_DIR + ' is specified. For additional options refer to the documentation.')

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_GROUND_TRUTH, args.print_ground_truth,
                                                 self.PRINT_GROUND_TRUTH_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options=options)]
        return []

    def __create_arff_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_GROUND_TRUTH, args.store_ground_truth,
                                                 self.STORE_GROUND_TRUTH_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            return [ArffFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_arff_file_sinks(args)

        if sinks:
            writer = GroundTruthWriter().add_sinks(*sinks)
            experiment.add_prediction_output_writers(writer)
