"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read algorithmic parameters from a source.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.cli import Argument
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.parameters.reader import ParameterReader
from mlrl.testbed.experiments.input.sources.source_csv import CsvFileSource
from mlrl.testbed.extensions.extension import Extension


class ParameterInputExtension(Extension):
    """
    An extension that configures the functionality to read algorithmic parameters from a source.
    """

    def get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_arguments`
        """
        return [
            Argument.string(
                '--parameter-load-dir',
                help='The path to the directory from which parameter to be used by the algorithm should be loaded.',
            ),
        ]

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        parameter_load_dir = args.parameter_load_dir

        if parameter_load_dir:
            reader = ParameterReader(CsvFileSource(parameter_load_dir))
            experiment_builder.add_input_readers(reader)
