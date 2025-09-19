"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read models parameters from one or several sources.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.extension import InputExtension
from mlrl.testbed.experiments.input.model.reader import ModelReader
from mlrl.testbed.experiments.input.sources.source_pickle import PickleFileSource
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, RunMode, SingleMode

from mlrl.util.cli import Argument, BoolArgument, PathArgument


class ModelInputExtension(Extension):
    """
    An extension that configures the functionality to read models from one or several sources.
    """

    MODEL_LOAD_DIR = PathArgument(
        '--model-load-dir',
        default='models',
        description='The path to the directory from which models should be loaded.',
    )

    LOAD_MODELS = BoolArgument(
        '--load-models',
        default=False,
        description='Whether models should be loaded from input files or not.',
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(InputExtension(), *dependencies)

    @override
    def _get_arguments(self, _: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.MODEL_LOAD_DIR, self.LOAD_MODELS}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        model_load_dir = self.MODEL_LOAD_DIR.get_value(args)

        if model_load_dir and self.LOAD_MODELS.get_value(args):
            reader = ModelReader(PickleFileSource(model_load_dir))
            experiment_builder.add_input_readers(reader)

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, RunMode}
