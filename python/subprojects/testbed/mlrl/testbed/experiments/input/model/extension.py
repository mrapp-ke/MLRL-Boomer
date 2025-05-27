"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read models parameters from a source.
"""
from argparse import ArgumentParser, Namespace

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.model.reader import ModelReader
from mlrl.testbed.experiments.input.sources.source_pickle import PickleFileSource
from mlrl.testbed.extensions.extension import Extension


class ModelInputExtension(Extension):
    """
    An extension that configures the functionality to read models from a source.
    """

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument('--model-load-dir',
                                     type=str,
                                     help='The path to the directory from which models should be loaded.')

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        model_load_dir = args.model_load_dir

        if model_load_dir:
            reader = ModelReader(PickleFileSource(model_load_dir))
            experiment.add_input_readers(reader)
