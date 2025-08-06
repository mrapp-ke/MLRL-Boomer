"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow to run experiments via the Slurm Workload Manager.
"""
from argparse import Namespace
from typing import override

from mlrl.testbed.experiments.recipe import Recipe
from mlrl.testbed.modes.mode_batch import Batch, BatchMode


class SlurmRunner(BatchMode.Runner):
    """
    A `BatchMode.Runner` that allows to run experiments via the Slurm Workload manager.
    """

    def __init__(self):
        super().__init__(name='slurm')

    # pylint: disable=unused-argument
    @override
    def run_batch(self, args: Namespace, batch: Batch, recipe: Recipe):
        """
        See :func:`mlrl.testbed.modes.mode_batch.BatchMode.Runner.run_batch`
        """
        print('Hello there')
