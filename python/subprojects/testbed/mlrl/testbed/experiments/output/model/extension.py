"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write models to one or several sinks.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.model.writer import ModelWriter
from mlrl.testbed.experiments.output.sinks.sink_pickle import PickleFileSink
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, StringArgument


class ModelOutputExtension(Extension):
    """
    An extension that configures the functionality to write models to one or several sinks.
    """

    PARAM_MODEL_SAVE_DIR = '--model-save-dir'

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            StringArgument(
                self.PARAM_MODEL_SAVE_DIR,
                help='The path to the directory where models should be saved.',
            ),
        ]

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        model_save_dir = args.model_save_dir

        if model_save_dir:
            pickle_sink = PickleFileSink(directory=model_save_dir, create_directory=args.create_output_dir)
            writer = ModelWriter().add_sinks(pickle_sink)
            experiment_builder.add_post_training_output_writers(writer)
