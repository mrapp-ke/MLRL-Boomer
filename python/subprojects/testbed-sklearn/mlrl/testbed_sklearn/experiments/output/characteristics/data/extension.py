"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write characteristics of tabular datasets to one or several
sinks.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics import OutputCharacteristics
from mlrl.testbed_sklearn.experiments.output.characteristics.data.characteristics_data import DataCharacteristics

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.sources import CsvFileSource
from mlrl.testbed.experiments.output.arguments import OutputArguments, ResultDirectoryArguments
from mlrl.testbed.experiments.output.extension import OutputExtension, ResultDirectoryExtension
from mlrl.testbed.experiments.output.sinks import CsvFileSink, LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode, SingleMode
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

    SAVE_DATA_CHARACTERISTICS = BoolArgument(
        '--save-data-characteristics',
        description='Whether the characteristics of the training data should be written to output files or not.',
        true_options={
            DataCharacteristics.OPTION_EXAMPLES, DataCharacteristics.OPTION_FEATURES,
            DataCharacteristics.OPTION_NUMERICAL_FEATURES, DataCharacteristics.OPTION_NOMINAL_FEATURES,
            DataCharacteristics.OPTION_FEATURE_DENSITY, DataCharacteristics.OPTION_FEATURE_SPARSITY,
            OutputCharacteristics.OPTION_OUTPUTS, OutputCharacteristics.OPTION_OUTPUT_DENSITY,
            OutputCharacteristics.OPTION_OUTPUT_SPARSITY, OutputCharacteristics.OPTION_LABEL_IMBALANCE_RATIO,
            OutputCharacteristics.OPTION_LABEL_CARDINALITY, OutputCharacteristics.OPTION_DISTINCT_LABEL_VECTORS,
            OPTION_DECIMALS
        },
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), ResultDirectoryExtension(), *dependencies)

    @override
    def _get_arguments(self, _: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PRINT_DATA_CHARACTERISTICS, self.SAVE_DATA_CHARACTERISTICS}

    def __configure_log_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_data_characteristics, options = self.PRINT_DATA_CHARACTERISTICS.get_value_and_options(args,
                                                                                                    default=print_all)

        if print_data_characteristics:
            experiment_builder.data_characteristics_writer.add_sinks(
                LogSink(options=options, source_factory=CsvFileSource))

    def __configure_csv_file_sink(self, args: Namespace, experiment_builder: Experiment.Builder):
        save_all = OutputArguments.SAVE_ALL.get_value(args)
        save_data_characteristics, options = self.SAVE_DATA_CHARACTERISTICS.get_value_and_options(args,
                                                                                                  default=save_all)
        base_dir = OutputArguments.BASE_DIR.get_value(args)
        result_directory = ResultDirectoryArguments.RESULT_DIR.get_value(args)

        if save_data_characteristics and base_dir and result_directory:
            create_directory = OutputArguments.CREATE_DIRS.get_value(args)
            experiment_builder.data_characteristics_writer.add_sinks(
                CsvFileSink(directory=base_dir / result_directory, create_directory=create_directory, options=options))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_csv_file_sink(args, experiment_builder)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, ReadMode, RunMode}
