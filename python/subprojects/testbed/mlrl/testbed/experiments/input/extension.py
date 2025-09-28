"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to read input data from one or several sources.
"""
from argparse import Namespace
from typing import Set, Type, override

from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.arguments import InputArguments
from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes import BatchMode, Mode, ReadMode, RunMode, SingleMode

from mlrl.util.cli import Argument


class InputExtension(Extension):
    """
    An extension that configures the functionality to read input data from one or several sources.
    """

    @override
    def _get_arguments(self, _: Mode) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return {InputArguments.EXIT_ON_MISSING_INPUT}

    @override
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder, _: Mode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_experiment`
        """
        experiment_builder.set_exit_on_missing_input(InputArguments.EXIT_ON_MISSING_INPUT.get_value(args))

    @override
    def get_supported_modes(self) -> Set[Type[Mode]]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.get_supported_modes`
        """
        return {SingleMode, BatchMode, ReadMode, RunMode}
