"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing a single experiment.
"""
import logging as log

from argparse import Namespace
from typing import List, Set, override

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.output.arguments import OutputArguments
from mlrl.testbed.experiments.output.policies import OutputExistsPolicy
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.experiments.state import ExperimentMode
from mlrl.testbed.modes.mode import Mode
from mlrl.testbed.modes.util import OutputUtil

from mlrl.util.cli import Argument, CommandLineInterface


class SingleMode(Mode):
    """
    A mode of operation that performs a single experiment.
    """

    @staticmethod
    def __should_experiment_be_cancelled(args: Namespace, recipe: Recipe, command: Command) -> bool:
        if OutputArguments.IF_OUTPUTS_EXIST.get_value(args) == OutputExistsPolicy.CANCEL:
            log.info('Checking if output files do already exist...')
            base_dir = OutputArguments.BASE_DIR.get_value(args)
            output_util = OutputUtil(args=args,
                                     recipe=recipe,
                                     command=command,
                                     input_directory=base_dir,
                                     file_sinks_only=True)

            if output_util.check_if_output_files_exist():
                return True

        return False

    @override
    def configure_arguments(self, cli: CommandLineInterface, extension_arguments: List[Argument],
                            algorithmic_arguments: List[Argument]):
        cli.add_arguments(*extension_arguments, *algorithmic_arguments)

    @override
    def run_experiment(self, extension_arguments: Set[Argument], algorithmic_arguments: Set[Argument], args: Namespace,
                       recipe: Recipe):
        command = Command.from_argv()

        if self.__should_experiment_be_cancelled(args=args, recipe=recipe, command=command):
            log.info(
                'Cancelling experiment, because all output files do already exist. Use the argument "%s %s" to '
                + 'force-run the experiment.', OutputArguments.IF_OUTPUTS_EXIST.name, OutputExistsPolicy.OVERWRITE)
        else:
            recipe.create_experiment_builder(experiment_mode=self.to_enum(), args=args, command=command).run(args)

    @override
    def to_enum(self) -> ExperimentMode:
        return ExperimentMode.SINGLE
