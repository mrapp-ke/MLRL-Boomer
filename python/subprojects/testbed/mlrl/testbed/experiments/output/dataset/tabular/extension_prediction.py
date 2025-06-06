"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write predictions to one or several sinks.
"""
from argparse import Namespace
from typing import List, Set

from mlrl.testbed_arff.experiments.output.sinks.sink_arff import ArffFileSink

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.dataset.tabular.writer_prediction import PredictionWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS

from mlrl.util.cli import Argument, BoolArgument


class PredictionExtension(Extension):
    """
    An extension that configures the functionality to write predictions to one or several sinks.
    """

    PRINT_PREDICTIONS = BoolArgument(
        '--print-predictions',
        default=False,
        description='Whether predictions should be printed on the console or not.',
        true_options={OPTION_DECIMALS},
    )

    STORE_PREDICTIONS = BoolArgument(
        '--store-predictions',
        default=False,
        description='Whether predictions should be written into output files or not. Does only have an effect, if the '
        + 'argument ' + OutputExtension.OUTPUT_DIR.name + ' is specified.',
        true_options={OPTION_DECIMALS},
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_PREDICTIONS, self.STORE_PREDICTIONS}

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.PRINT_PREDICTIONS.get_value(args)

        if value or (value is None and args.print_all):
            return [LogSink(options=options)]
        return []

    def __create_arff_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.STORE_PREDICTIONS.get_value(args)
        output_dir = OutputExtension.OUTPUT_DIR.get_value(args)

        if value and output_dir:
            return [ArffFileSink(directory=output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_arff_file_sinks(args)

        if sinks:
            writer = PredictionWriter().add_sinks(*sinks)
            experiment_builder.add_prediction_output_writers(writer)
