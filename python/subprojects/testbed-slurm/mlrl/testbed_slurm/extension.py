"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that allow configuring the functionality to run experiments via the Slurm Workload Manager.
"""
from argparse import Namespace
from typing import Set, override

from mlrl.testbed_slurm.runner import SlurmRunner

from mlrl.testbed.extensions.extension import Extension
from mlrl.testbed.modes.mode_batch import BatchMode

from mlrl.util.cli import Argument


class SlurmExtension(Extension):
    """
    An extension that configures the functionality to run experiments via the Slurm Workload Manager.
    """

    @override
    def configure_batch_mode(self, _: Namespace, batch_mode: BatchMode):
        """
        See :func:`mlrl.testbed.extensions.extension.Extension.configure_batch_mode`
        """
        batch_mode.add_runner(SlurmRunner())

    @override
    def _get_arguments(self) -> Set[Argument]:
        """
        See :func:`mlrl.testbed.extensions.extension.Extension._get_arguments`
        """
        return set()
