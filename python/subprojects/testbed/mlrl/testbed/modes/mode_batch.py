"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing multiple experiments.
"""
from argparse import Namespace
from typing import override

from mlrl.testbed.experiments import Experiment
from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import CommandLineInterface, FlagArgument, StringArgument


class BatchExperimentMode(Mode):
    """
    A mode of operation that performs multiple experiments.
    """

    CONFIG_FILE = StringArgument(
        '-c',
        '--config',
        required=True,
        description='An absolute or relative path to a YAML file that configures the batch of experiments to be run.',
    )

    LIST_COMMANDS = FlagArgument(
        '--list',
        description='Lists the commands for running individual experiments instead of executing them.',
    )

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        cli.add_arguments(self.CONFIG_FILE, self.LIST_COMMANDS)

    @override
    def run_experiment(self, _: Namespace, experiment_builder_factory: Experiment.Builder.Factory):
        pass
