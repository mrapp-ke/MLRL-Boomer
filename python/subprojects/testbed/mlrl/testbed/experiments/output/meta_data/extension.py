"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write meta-data to one or several sinks.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.meta_data.arguments import MetaDataArguments
from mlrl.testbed.experiments.output.sinks import LogSink, YamlFileSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, RunMode, SingleMode

from mlrl.util.cli import AUTO, Argument
from mlrl.util.options import BooleanOption


class MetaDataExtension(Extension):
    """
    An extension that configures the functionality to write meta-data to one or several sinks.
    """

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    @override
    def _get_arguments(self, _: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {MetaDataArguments.PRINT_META_DATA, MetaDataArguments.SAVE_META_DATA}

    @staticmethod
    def __configure_log_sink(args: Namespace, experiment_builder: Experiment.Builder):
        print_meta_data = MetaDataArguments.PRINT_META_DATA.get_value(args)

        if print_meta_data:
            experiment_builder.meta_data_writer.add_sinks(LogSink())

    @staticmethod
    def __configure_yaml_file_sink(args: Namespace, experiment_builder: Experiment.Builder):
        save_meta_data = MetaDataArguments.SAVE_META_DATA.get_value(args)

        if save_meta_data == BooleanOption.TRUE or (save_meta_data == AUTO
                                                    and experiment_builder.has_output_file_writers):
            base_dir = OutputArguments.BASE_DIR.get_value(args)

            if base_dir:
                create_directory = OutputArguments.CREATE_DIRS.get_value(args)
                experiment_builder.meta_data_writer.add_sinks(
                    YamlFileSink(directory=base_dir, create_directory=create_directory))

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        self.__configure_log_sink(args, experiment_builder)
        self.__configure_yaml_file_sink(args, experiment_builder)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, RunMode}
