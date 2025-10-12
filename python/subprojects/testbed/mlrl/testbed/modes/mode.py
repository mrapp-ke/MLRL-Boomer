"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing different modes of operation.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List

from mlrl.testbed.experiments.recipe import Recipe

from mlrl.util.cli import Argument, CommandLineInterface, SetArgument


class Mode(ABC):
    """
    An abstract base class for all modes of operation.
    """

    MODE_SINGLE = 'single'

    MODE_BATCH = 'batch'

    MODE_RUN = 'run'

    MODE = SetArgument(
        '--mode',
        values={MODE_SINGLE, MODE_BATCH, MODE_RUN},
        description='The mode of operation to be used.',
        default=MODE_SINGLE,
    )

    @abstractmethod
    def configure_arguments(self, cli: CommandLineInterface, extension_arguments: List[Argument],
                            algorithmic_arguments: List[Argument]):
        """
        Must be implemented by subclasses in order to configure the command line interface according to the mode of
        operation.

        :param cli:                     The command line interface to be configured
        :param extension_arguments:     The arguments that should be added to the command line interface according to
                                        the registered extensions
        :param algorithmic_arguments:   The arguments that should be added to the command line interface for configuring
                                        the algorithm's hyperparameters
        """

    @abstractmethod
    def run_experiment(self, args: Namespace, recipe: Recipe):
        """
        Must be implemented by subclasses in order to run an experiment according to the command line arguments
        specified by the user.

        :param args:    The command line arguments specified by the user
        :param recipe:  A `Recipe` that provides access to the ingredients that are needed for setting up experiments
        """
