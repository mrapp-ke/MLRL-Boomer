"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing a single experiment.
"""
from argparse import Namespace
from typing import override

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import CommandLineInterface


class SingleExperimentMode(Mode):
    """
    A mode of operation that performs a single experiment.
    """

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        pass

    @override
    def run_experiment(self, args: Namespace, experiment_builder_factory: Experiment.Builder.Factory):
        experiment_builder = experiment_builder_factory(args)
        experiment_builder.run()
