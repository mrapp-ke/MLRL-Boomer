"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes that implement a mode of operation for performing a single experiment.
"""
from typing import override

from mlrl.testbed.modes.mode import Mode

from mlrl.util.cli import CommandLineInterface


class SingleExperimentMode(Mode):
    """
    A mode of operation that performs a single experiment.
    """

    @override
    def configure_arguments(self, cli: CommandLineInterface):
        pass
