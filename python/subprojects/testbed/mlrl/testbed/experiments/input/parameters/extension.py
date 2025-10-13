"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read algorithmic parameters from one or several sources.
"""
from argparse import Namespace
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.extension import InputExtension
from mlrl.testbed.experiments.input.parameters.reader import ParameterReader
from mlrl.testbed.experiments.input.sources.source_csv import CsvFileSource
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument, BoolArgument, PathArgument


class ParameterInputExtension(Extension):
    """
    An extension that configures the functionality to read algorithmic parameters from one or several sources.
    """

    PARAMETER_LOAD_DIR = PathArgument(
        '--parameter-load-dir',
        default='parameters',
        description='The path to the directory from which parameters to be used by the algorithm should be loaded.',
    )

    LOAD_PARAMETERS = BoolArgument(
        '--load-parameters',
        default=False,
        description='Whether parameters should be loaded from input files or not.',
    )

    def __init__(self, *dependencies: Extension):
        """
        :param dependencies: Other extensions, this extension depends on
        """
        super().__init__(InputExtension(), *dependencies)

    @override
    def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {self.PARAMETER_LOAD_DIR, self.LOAD_PARAMETERS}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: ExperimentMode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        parameter_load_dir = self.PARAMETER_LOAD_DIR.get_value(args)

        if parameter_load_dir and self.LOAD_PARAMETERS.get_value(args):
            reader = ParameterReader(CsvFileSource(parameter_load_dir))
            experiment_builder.add_input_readers(reader)

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.SINGLE, ExperimentMode.BATCH, ExperimentMode.RUN}
