"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing a single experiment.
"""
from argparse import Namespace
from typing import List, Set, override

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import Argument, CommandLineInterface


class SingleMode(Mode):
    """
    A mode of operation that performs a single experiment.
    """

    @override
    def configure_arguments(self, cli: CommandLineInterface, extension_arguments: List[Argument],
                            algorithmic_arguments: List[Argument]):
        cli.add_arguments(*extension_arguments, *algorithmic_arguments)

    @override
    def run_experiment(self, extension_arguments: Set[Argument], algorithmic_arguments: Set[Argument], args: Namespace,
                       recipe: Recipe):
        command = Command.from_argv()
        experiment_builder = recipe.create_experiment_builder(experiment_mode=self.to_enum(),
                                                              args=args,
                                                              command=command)
        experiment_builder.run(args)

    @override
    def to_enum(self) -> ExperimentMode:
        return ExperimentMode.SINGLE
