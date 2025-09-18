"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write models to one or several sinks.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.arguments import OutputArguments
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.model.arguments import ModelOutputDirectoryArguments
from mlrl.testbed.experiments.output.sinks import PickleFileSink
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode, SingleMode

from mlrl.util.cli import Argument, BoolArgument


class ModelOutputExtension(Extension):
    """
    An extension that configures the functionality to write models to one or several sinks.
    """

    SAVE_MODELS = BoolArgument(
        '--save-models',
        description='Whether models should be saved to output files or not.',
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
        return {self.SAVE_MODELS}

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, ReadMode, RunMode}


class ModelOutputDirectoryExtension(Extension):
    """
    An extension that configures the directory to which models should be written.
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
        return set() if isinstance(mode, BatchMode) else {ModelOutputDirectoryArguments.MODEL_SAVE_DIR}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        save_all = OutputArguments.SAVE_ALL.get_value(args)

        if ModelOutputExtension.SAVE_MODELS.get_value(args, default=save_all):
            base_dir = OutputArguments.BASE_DIR.get_value(args)
            model_save_dir = ModelOutputDirectoryArguments.MODEL_SAVE_DIR.get_value(args)

            if base_dir and model_save_dir:
                create_directory = OutputArguments.CREATE_DIRS.get_value(args)
                experiment_builder.model_writer.add_sinks(
                    PickleFileSink(directory=base_dir / model_save_dir, create_directory=create_directory))

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, ReadMode, RunMode}
