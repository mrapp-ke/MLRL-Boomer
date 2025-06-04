"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read algorithmic parameters from a source.
"""
from argparse import Namespace
from typing import List

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.parameters.reader import ParameterReader
from mlrl.testbed.experiments.input.sources.source_csv import CsvFileSource
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, StringArgument


class ParameterInputExtension(Extension):
    """
    An extension that configures the functionality to read algorithmic parameters from a source.
    """

    PARAMETER_LOAD_DIR = StringArgument(
        '--parameter-load-dir',
        description='The path to the directory from which parameter to be used by the algorithm should be loaded.',
    )

    def _get_arguments(self) -> List[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return [self.PARAMETER_LOAD_DIR]

    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        parameter_load_dir = self.PARAMETER_LOAD_DIR.get_value(args)

        if parameter_load_dir:
            reader = ParameterReader(CsvFileSource(parameter_load_dir))
            experiment_builder.add_input_readers(reader)
