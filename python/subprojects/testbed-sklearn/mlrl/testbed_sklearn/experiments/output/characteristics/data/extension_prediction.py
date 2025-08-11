"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of binary predictions to one or
several sinks.
"""
from argparse import Namespace
from pathlib import Path
from typing import Set, override

from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import OutputCharacteristics
from mlrl.testbed_sklearn.experiments.prediction.extension import PredictionTypeExtension

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.experiments.prediction_type import PredictionType
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.cli import Argument, BoolArgument


class PredictionCharacteristicsExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of binary predictions to one or several
    sinks.
    """

    PRINT_PREDICTION_CHARACTERISTICS = BoolArgument(
        '--print-prediction-characteristics',
        description='Whether the characteristics of binary predictions should be printed on the console or not. Does '
        + 'only have an effect if the argument ' + PredictionTypeExtension.PREDICTION_TYPE.name + ' is set to '
        + PredictionType.BINARY.value + '.',
        true_options={
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
    )

    SAVE_PREDICTION_CHARACTERISTICS = BoolArgument(
        '--save-prediction-characteristics',
        description='Whether the characteristics of binary predictions should be written to output files or not. Does '
        + 'only have an effect if the argument ' + PredictionTypeExtension.PREDICTION_TYPE.name + ' is set to '
        + PredictionType.BINARY.value + ' and if the argument ' + ResultDirectoryArguments.RESULT_DIR.name + ' is '
        + 'specified.',
        true_options={
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
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_PREDICTION_CHARACTERISTICS, self.SAVE_PREDICTION_CHARACTERISTICS}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_prediction_characteristics, options = self.PRINT_PREDICTION_CHARACTERISTICS.get_value(args,
                                                                                                    default=print_all)

        if print_prediction_characteristics:
            experiment_builder.prediction_characteristics_writer.add_sinks(LogSink(options))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        save_all = OutputArguments.SAVE_ALL.get_value(args)
        save_prediction_characteristics, options = self.SAVE_PREDICTION_CHARACTERISTICS.get_value(args,
                                                                                                  default=save_all)
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if save_prediction_characteristics and result_directory:
            create_directory = OutputArguments.CREATE_DIRS.get_value(args)
            experiment_builder.prediction_characteristics_writer.add_sinks(
                CsvFileSink(directory=Path(result_directory), create_directory=create_directory, options=options))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_csv_file_sink(args, experiment_builder)
