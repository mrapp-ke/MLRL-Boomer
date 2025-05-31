"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of tabular datasets to one or several
sinks.
"""
from argparse import Namespace
from typing import Dict, List, Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.characteristics.data.tabular.characteristics import OutputCharacteristics
from mlrl.testbed.experiments.output.characteristics.data.tabular.characteristics_data import DataCharacteristics
from mlrl.testbed.experiments.output.characteristics.data.tabular.writer_data import DataCharacteristicsWriter
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink import Sink
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.cli import Argument, BoolArgument
from mlrl.util.options import BooleanOption, parse_param_and_options


class TabularDataCharacteristicExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of tabular datasets to one or several sinks.
    """

    PARAM_PRINT_DATA_CHARACTERISTICS = '--print-data-characteristics'

    PRINT_DATA_CHARACTERISTICS_VALUES: Dict[str, Set[str]] = {
        BooleanOption.TRUE.value: {
            DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
            DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
            DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
        BooleanOption.FALSE.value: {}
    }

    PARAM_STORE_DATA_CHARACTERISTICS = '--store-data-characteristics'

    STORE_DATA_CHARACTERISTICS_VALUES = PRINT_DATA_CHARACTERISTICS_VALUES

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            BoolArgument(
                self.PARAM_PRINT_DATA_CHARACTERISTICS,
                default=False,
                help='Whether the characteristics of the training data should be printed on the console or not.',
                true_options={
                    DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
                    DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
                    DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
                    OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
                    OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
                    OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
                    OPTION_DECIMALS, OPTION_PERCENTAGE
                },
            ),
            BoolArgument(
                self.PARAM_STORE_DATA_CHARACTERISTICS,
                default=False,
                help='Whether the characteristics of the training data should be written into output files or not. '
                + 'Does only have an effect if the argument ' + OutputExtension.PARAM_OUTPUT_DIR + ' is specified.',
                true_options={
                    DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
                    DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
                    DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
                    OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
                    OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
                    OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
                    OPTION_DECIMALS, OPTION_PERCENTAGE
                },
            ),
        ]

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_PRINT_DATA_CHARACTERISTICS, args.print_data_characteristics,
                                                 self.PRINT_DATA_CHARACTERISTICS_VALUES)

        if (not value and args.print_all) or value == BooleanOption.TRUE.value:
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = parse_param_and_options(self.PARAM_STORE_DATA_CHARACTERISTICS, args.store_data_characteristics,
                                                 self.STORE_DATA_CHARACTERISTICS_VALUES)

        if ((not value and args.store_all) or value == BooleanOption.TRUE.value) and args.output_dir:
            return [CsvFileSink(directory=args.output_dir, create_directory=args.create_output_dir, options=options)]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = DataCharacteristicsWriter().add_sinks(*sinks)
            experiment_builder.add_pre_training_output_writers(writer)
