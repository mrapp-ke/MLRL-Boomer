"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read models parameters from a source.
"""
from argparse import Namespace
from typing import Set

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.model.reader import ModelReader
from mlrl.testbed.experiments.input.sources.source_pickle import PickleFileSource
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, StringArgument


class ModelInputExtension(Extension):
    """
    An extension that configures the functionality to read models from a source.
    """

    MODEL_LOAD_DIR = StringArgument(
        '--model-load-dir',
        description='The path to the directory from which models should be loaded.',
    )

    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.MODEL_LOAD_DIR}

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        model_load_dir = self.MODEL_LOAD_DIR.get_value(args)

        if model_load_dir:
            reader = ModelReader(PickleFileSource(model_load_dir))
            experiment_builder.add_input_readers(reader)
