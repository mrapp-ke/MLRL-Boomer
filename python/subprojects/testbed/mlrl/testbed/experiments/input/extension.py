"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read input data from one or several sources.
"""
from argparse import Namespace
from typing import Set, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.arguments import InputArguments
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.extensions.extension import Extension

from mlrl.util.cli import Argument


class InputExtension(Extension):
    """
    An extension that configures the functionality to read input data from one or several sources.
    """

    @override
    def _get_arguments(self, _: ExperimentMode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {InputArguments.IF_INPUT_MISSING}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: ExperimentMode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.set_missing_input_policy(InputArguments.IF_INPUT_MISSING.get_value(args))

    @override
    def get_supported_modes(self) -> Set[ExperimentMode]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {ExperimentMode.SINGLE, ExperimentMode.BATCH, ExperimentMode.READ, ExperimentMode.RUN}
