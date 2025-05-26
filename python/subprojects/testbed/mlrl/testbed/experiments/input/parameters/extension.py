"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read algorithmic parameters from a source.
"""
from argparse import ArgumentParser, Namespace

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.parameters.reader import ParameterReader
from mlrl.testbed.experiments.input.sources.source_csv import CsvFileSource
from mlrl.testbed.extensions.extension import Extension


class ParameterInputExtension(Extension):
    """
    An extension that configures the functionality to read algorithmic parameters from a source.
    """

    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_arguments`
        """
        argument_parser.add_argument(
            '--parameter-load-dir',
            type=str,
            help='The path to the directory from which parameter to be used by the algorithm should be loaded.')

    def configure_experiment(self, args: Namespace, experiment: Experiment):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        parameter_load_dir = args.parameter_load_dir

        if parameter_load_dir:
            reader = ParameterReader(CsvFileSource(parameter_load_dir))
            experiment.add_input_readers(reader)
