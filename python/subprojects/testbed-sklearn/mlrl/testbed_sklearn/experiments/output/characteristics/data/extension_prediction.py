"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of binary predictions to one or
several sinks.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import OutputCharacteristics
from mlrl.testbed_sklearn.experiments.prediction.extension import PredictionTypeExtension

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode, SingleMode
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.cli import Argument, BoolArgument


class PredictionCharacteristicsExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of binary predictions to one or several
    sinks.
    """

    @staticmethod
    def __create_argument_print_prediction_characteristics(mode: Mode) -> BoolArgument:
        return BoolArgument(
            '--print-prediction-characteristics',
            description='Whether the characteristics of binary predictions should be printed on the console or not. '
            + 'Does only have an effect if the argument ' + PredictionTypeExtension.PREDICTION_TYPE.name + ' is set to '
            + PredictionType.BINARY.value + '.',
            true_options=None if isinstance(mode, ReadMode) else {
                OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
                OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
                OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
                OPTION_DECIMALS, OPTION_PERCENTAGE
            },
        )

    @staticmethod
    def __create_argument_save_prediction_characteristics(mode: Mode) -> BoolArgument:
        return BoolArgument(
            '--save-prediction-characteristics',
            description='Whether the characteristics of binary predictions should be written to output files or not. '
            + 'Does only have an effect if the argument ' + PredictionTypeExtension.PREDICTION_TYPE.name + ' is set to '
            + PredictionType.BINARY.value + ' and if the argument ' + ResultDirectoryArguments.RESULT_DIR.name + ' is '
            + 'specified.',
            true_options=None if isinstance(mode, ReadMode) else {
                OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
                OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
                OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
                OPTION_DECIMALS, OPTION_PERCENTAGE
            },
        )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self, mode: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {
            self.__create_argument_print_prediction_characteristics(mode),
            self.__create_argument_save_prediction_characteristics(mode),
        }

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder, mode: Mode):
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_prediction_characteristics, options = self.__create_argument_print_prediction_characteristics(
            mode).get_value_and_options(args, default=print_all)

        if print_prediction_characteristics:
            experiment_builder.prediction_characteristics_writer.add_sinks(LogSink(options))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder, mode: Mode):
        save_all = OutputArguments.SAVE_ALL.get_value(args)
        save_prediction_characteristics, options = self.__create_argument_save_prediction_characteristics(
            mode).get_value_and_options(args, default=save_all)
        base_dir = OutputArguments.BASE_DIR.get_value(args)
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if save_prediction_characteristics and base_dir and result_directory:
            create_directory = OutputArguments.CREATE_DIRS.get_value(args)
            experiment_builder.prediction_characteristics_writer.add_sinks(
                CsvFileSink(directory=base_dir / result_directory, create_directory=create_directory, options=options))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, mode: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder, mode)
        self.__configure_csv_file_sink(args, experiment_builder, mode)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, ReadMode, RunMode}
