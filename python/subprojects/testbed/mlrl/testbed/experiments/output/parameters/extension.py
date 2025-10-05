"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write algorithmic parameters to one or several sinks.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.parameters.arguments import ParameterOutputDirectoryArguments
from mlrl.testbed.experiments.output.sinks import CsvFileSink, LogSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, RunMode, SingleMode

from mlrl.util.cli import Argument, BoolArgument


class ParameterOutputExtension(Extension):
    """
    An extension that configures the functionality to write algorithmic parameters to one or several sinks.
    """

    PRINT_PARAMETERS = BoolArgument(
        '--print-parameters',
        description='Whether the parameter setting should be printed on the console or not.',
    )

    SAVE_PARAMETERS = BoolArgument(
        '--save-parameters',
        description='Whether the parameter setting should be written to output files or not.',
    )

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
        return {self.PRINT_PARAMETERS, self.SAVE_PARAMETERS}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        print_all = OutputArguments.PRINT_ALL.get_value(args)
        print_parameters = self.PRINT_PARAMETERS.get_value(args, default=print_all)

        if print_parameters:
            experiment_builder.parameter_writer.add_sinks(LogSink())

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, RunMode}


class ParameterOutputDirectoryExtension(Extension):
    """
    An extension that configures the directory to which algorithmic parameters should be written.
    """

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    @override
    def _get_arguments(self, mode: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return set() if isinstance(mode, BatchMode) else {ParameterOutputDirectoryArguments.PARAMETER_SAVE_DIR}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        save_all = OutputArguments.SAVE_ALL.get_value(args)

        if ParameterOutputExtension.SAVE_PARAMETERS.get_value(args, default=save_all):
            base_dir = OutputArguments.BASE_DIR.get_value(args)
            parameter_save_dir = ParameterOutputDirectoryArguments.PARAMETER_SAVE_DIR.get_value(args)

            if base_dir and parameter_save_dir:
                create_directory = OutputArguments.CREATE_DIRS.get_value(args)
                experiment_builder.parameter_writer.add_sinks(
                    CsvFileSink(directory=base_dir / parameter_save_dir, create_directory=create_directory))

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, RunMode}
