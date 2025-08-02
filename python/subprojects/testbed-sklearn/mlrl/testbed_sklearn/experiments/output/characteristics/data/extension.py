"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of tabular datasets to one or several
sinks.
"""
from argparse import Namespace
from typing import Set

from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import OutputCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics_data import DataCharacteristics

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink_csv import CsvFileSink
from mlrl.testbed.experiments.output.sinks.sink_log import LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.util.format import OPTION_DECIMALS, OPTION_PERCENTAGE

from mlrl.util.cli import Argument, BoolArgument


class TabularDataCharacteristicExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of tabular datasets to one or several sinks.
    """

    PRINT_DATA_CHARACTERISTICS = BoolArgument(
        '--print-data-characteristics',
        description='Whether the characteristics of the training data should be printed on the console or not.',
        true_options={
            DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
            DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
            DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS, OPTION_PERCENTAGE
        },
    )

    STORE_DATA_CHARACTERISTICS = BoolArgument(
        '--store-data-characteristics',
        description='Whether the characteristics of the training data should be written into output files or not. '
        + 'Does only have an effect if the argument ' + OutputExtension.OUTPUT_DIR.name + ' is specified.',
        true_options={
            DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
            DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
            DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
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
        super().__init__(OutputExtension(), *dependencies)

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_DATA_CHARACTERISTICS, self.STORE_DATA_CHARACTERISTICS}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputExtension.PRINT_ALL.get_value(args)
        print_data_characteristics, options = self.PRINT_DATA_CHARACTERISTICS.get_value(args, default=print_all)

        if print_data_characteristics:
            experiment_builder.data_characteristics_writer.add_sinks(LogSink(options))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        store_all = OutputExtension.STORE_ALL.get_value(args)
        store_data_characteristics, options = self.STORE_DATA_CHARACTERISTICS.get_value(args, default=store_all)
        output_directory = OutputExtension.OUTPUT_DIR.get_value(args)

        if store_data_characteristics and output_directory:
            create_output_directory = OutputExtension.CREATE_OUTPUT_DIR.get_value(args)
            experiment_builder.data_characteristics_writer.add_sinks(
                CsvFileSink(directory=output_directory, create_directory=create_output_directory, options=options))

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_csv_file_sink(args, experiment_builder)
