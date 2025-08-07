"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run experiments via the Slurm Workload Manager.
"""
import logging as log
import sys

from argparse import Namespace
from pathlib import Path
from typing import override
from uuid import uuid4

from tabulate import tabulate

from mlrl.testbed_slurm.arguments import SlurmArguments
from mlrl.testbed_slurm.sbatch import Sbatch

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.modes.mode_batch import Batch, BatchMode
from mlrl.testbed.util.io import open_writable_file


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
    def __write_sbatch_file(command: Command) -> Path:
        path = Path('sbatch_' + str(uuid4()).split('-', maxsplit=1)[0] + '.sh')

        with open_writable_file(path) as sbatch_file:
            sbatch_file.write('\n'.join([
                '#!/bin/sh',
                str(command),
            ]))

        return path

    def __submit_command(self, args: Namespace, command: Command):
        sbatch_file = SlurmRunner.__write_sbatch_file(command)

        if SlurmArguments.SAVE_SLURM_SCRIPTS.get_value(args):
            log.info('Slurm script saved to file "%s"', sbatch_file)
        else:
            job_name = sbatch_file.stem
            result = Sbatch().script(sbatch_file).run()
            sbatch_file.unlink()

            if result.ok:
                job_id = result.output.split(' ')[-1]
                log.info('Successfully submitted job:\n\n%s',
                         tabulate([['JOBID', job_id], ['NAME', job_name]], tablefmt='plain'))
            else:
                log.error('Submission to Slurm failed:\n%s', result.output)
                sys.exit(result.exit_code)

    def __init__(self):
        super().__init__(name='slurm')

    @override
    def run_batch(self, args: Namespace, batch: Batch, _: Recipe):
        """
        See :func:`mlrl.testbed.modes.mode_batch.BatchMode.Runner.run_batch`
        """
        num_experiments = len(batch)

        if self.__is_command_available():
            log.info('Submitting %s %s to Slurm...', num_experiments,
                     'experiments' if num_experiments > 1 else 'experiment')

            for i, command in enumerate(batch):
                log.info('\nSubmitting experiment (%s / %s): "%s"', i + 1, num_experiments, str(command))
                self.__submit_command(args, command)
