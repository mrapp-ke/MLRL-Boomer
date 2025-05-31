"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write ground truth to one or several sinks.
"""
from argparse import Namespace
from typing import Dict, List, Set

from mlrl.testbed_arff.experiments.output.sinks.sink_arff import ArffFileSink

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.dataset.tabular.writer_ground_truth import GroundTruthWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.cli import Argument, BoolArgument
from mlrl.util.options import BooleanOption, parse_param_and_options


class GroundTruthExtension(Extension):
    """
    An extension that configures the functionality to write ground truth to one or several sinks.
    """

    PARAM_PRINT_GROUND_TRUTH = '--print-ground-truth'

    PRINT_GROUND_TRUTH_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {OPTION_DECIMALS},
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_GROUND_TRUTH = '--store-ground-truth'

    STORE_GROUND_TRUTH_VALUES = PRINT_GROUND_TRUTH_VALUES

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            BoolArgument(
                self.PARAM_PRINT_GROUND_TRUTH,
                default=False,
                help='Whether the ground truth should be printed on the console or not.',
                true_options={OPTION_DECIMALS},
            ),
            BoolArgument(
                self.PARAM_STORE_GROUND_TRUTH,
                default=False,
                help='Whether the ground truth should be written into output files or not. Does only have an effect, '
                + 'if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is specified.',
                true_options={OPTION_DECIMALS},
            ),
        ]

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

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_arff_file_sinks(args)

        if sinks:
            writer = GroundTruthWriter().add_sinks(*sinks)
            experiment_builder.add_prediction_output_writers(writer)
