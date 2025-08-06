"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing a single experiment.
"""
import sys

from argparse import Namespace
from typing import override

from mlrl.testbed.command import ArgumentList, Command
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import CommandLineInterface


class SingleMode(Mode):
    """
    A mode of operation that performs a single experiment.
    """

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        pass

    @override
    def run_experiment(self, args: Namespace, recipe: Recipe):
        command = Command(module_name=sys.argv[1], argument_list=ArgumentList(sys.argv[2:]))
        experiment_builder = recipe.create_experiment_builder(args, command)
        experiment_builder.run()
