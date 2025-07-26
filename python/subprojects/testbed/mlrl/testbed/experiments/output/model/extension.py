"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write models to one or several sinks.
"""
from argparse import Namespace
from os import path
from pathlib import Path
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.extension import OutputExtension
from mlrl.testbed.experiments.output.sinks.sink_pickle import PickleFileSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument, StringArgument


class ModelOutputExtension(Extension):
    """
    An extension that configures the functionality to write models to one or several sinks.
    """

    MODEL_SAVE_DIR = StringArgument(
        '--model-save-dir',
        default='models',
        description='The path to the directory where models should be saved.',
        decorator=lambda args, value: path.join(OutputExtension.BASE_DIR.get_value(args), value),
    )

    SAVE_MODELS = BoolArgument(
        '--save-models',
        default=False,
        description='Whether models should be saved to output files or not.',
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(OutputExtension(), *dependencies)

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.MODEL_SAVE_DIR, self.SAVE_MODELS}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        if self.SAVE_MODELS.get_value(args):
            model_save_dir = self.MODEL_SAVE_DIR.get_value(args)

            if model_save_dir:
                create_directory = OutputExtension.CREATE_DIRS.get_value(args)
                experiment_builder.model_writer.add_sinks(
                    PickleFileSink(directory=Path(model_save_dir), create_directory=create_directory))
