"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write predictions to one or several sinks.
"""
from argparse import Namespace
from typing import Set

from mlrl.testbed_arff.experiments.output.sinks.sink_arff import ArffFileSink

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
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
        description='Whether predictions should be written into output files or not.',
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

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputExtension.PRINT_ALL.get_value(args)
        print_predictions, options = self.PRINT_PREDICTIONS.get_value(args, default=print_all)

        if print_predictions:
            experiment_builder.prediction_writer.add_sinks(LogSink(options=options))

    def __configure_arff_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        store_all = OutputExtension.STORE_ALL.get_value(args)
        store_predictions, options = self.STORE_PREDICTIONS.get_value(args, default=store_all)
        output_directory = OutputExtension.OUTPUT_DIR.get_value(args)

        if store_predictions and output_directory:
            create_output_directory = OutputExtension.CREATE_OUTPUT_DIR.get_value(args)
            experiment_builder.prediction_writer.add_sinks(
                ArffFileSink(directory=output_directory, create_directory=create_output_directory, options=options))

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_arff_file_sink(args, experiment_builder)
