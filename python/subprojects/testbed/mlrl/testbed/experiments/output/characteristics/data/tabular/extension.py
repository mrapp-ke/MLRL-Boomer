"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of tabular datasets to one or several
sinks.
"""
from argparse import Namespace
from typing import List, Set

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


class TabularDataCharacteristicExtension(Extension):
    """
    An extension that configures the functionality to write characteristics of tabular datasets to one or several sinks.
    """

    PRINT_DATA_CHARACTERISTICS = BoolArgument(
        '--print-data-characteristics',
        default=False,
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
        default=False,
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

    def __create_log_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.PRINT_DATA_CHARACTERISTICS.get_value(args,
                                                                   default=OutputExtension.PRINT_ALL.get_value(args))

        if value:
            return [LogSink(options)]
        return []

    def __create_csv_file_sinks(self, args: Namespace) -> List[Sink]:
        value, options = self.STORE_DATA_CHARACTERISTICS.get_value(args)
        output_dir = OutputExtension.OUTPUT_DIR.get_value(args, default=OutputExtension.STORE_ALL.get_value(args))

        if value and output_dir:
            return [
                CsvFileSink(directory=output_dir,
                            create_directory=OutputExtension.CREATE_OUTPUT_DIR.get_value(args),
                            options=options)
            ]
        return []

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        sinks = self.__create_log_sinks(args) + self.__create_csv_file_sinks(args)

        if sinks:
            writer = DataCharacteristicsWriter().add_sinks(*sinks)
            experiment_builder.add_pre_training_output_writers(writer)
