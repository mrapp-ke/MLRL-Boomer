"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to write models to one or several sinks.
"""
from argparse import ArgumentParser, Namespace

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.output.model.writer import ModelWriter
from mlrl.testbed.experiments.output.sinks.sink_pickle import PickleFileSink
from mlrl.testbed.extensions.extension import Extension


class ModelOutputExtension(Extension):
    """
    An extension that configures the functionality to write models to one or several sinks.
    """

    PARAM_MODEL_SAVE_DIR = '--model-save-dir'

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(self.PARAM_MODEL_SAVE_DIR,
                                     type=str,
                                     help='The path to the directory where models should be saved.')

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        model_save_dir = args.model_save_dir

        if model_save_dir:
            pickle_sink = PickleFileSink(directory=model_save_dir, create_directory=args.create_output_dir)
            writer = ModelWriter(exit_on_error=args.exit_on_error).add_sinks(pickle_sink)
            experiment.add_post_training_output_writers(writer)
