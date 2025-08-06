"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run experiments via the Slurm Workload Manager.
"""
import logging as log
import sys

from argparse import Namespace
from typing import override

from mlrl.testbed_slurm.sbatch import Sbatch

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.modes.mode_batch import Batch, BatchMode


class SlurmRunner(BatchMode.Runner):
    """
    A `BatchMode.Runner` that allows to run experiments via the Slurm Workload manager.
    """

    @staticmethod
    def __is_command_available() -> bool:
        sbatch = Sbatch()
        version = sbatch.version().run()

        if not version.ok:
            log.error('Command "%s" not found: %s', sbatch.command, version.output)
            return False

        return True

    @staticmethod
    def __submit_command(command: Command):
        result = Sbatch().wrap(str(command)).run()

        if result.ok:
            log.info('%s', result.output)
        else:
            log.error('Submission to Slurm failed:\n%s', result.output)
            sys.exit(result.exit_code)

    def __init__(self):
        super().__init__(name='slurm')

    # pylint: disable=unused-argument
    @override
    def run_batch(self, args: Namespace, batch: Batch, recipe: Recipe):
        """
        See :func:`mlrl.testbed.modes.mode_batch.BatchMode.Runner.run_batch`
        """
        num_experiments = len(batch)

        if self.__is_command_available():
            log.info('Submitting %s %s to Slurm...', num_experiments,
                     'experiments' if num_experiments > 1 else 'experiment')

            for i, command in enumerate(batch):
                log.info('\nSubmitting experiment (%s / %s): "%s"', i + 1, num_experiments, str(command))
                self.__submit_command(command)
