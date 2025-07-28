"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing different modes of operation.
"""
from abc import ABC, abstractmethod
from argparse import Namespace

from mlrl.testbed.experiments import Experiment

from mlrl.util.cli import CommandLineInterface, SetArgument


class Mode(ABC):
    """
    An abstract base class for all modes of operation.
    """

    MODE_SINGLE = 'single'

    MODE_BATCH = 'batch'

    MODE = SetArgument(
        '--mode',
        values={MODE_SINGLE, MODE_BATCH},
        description='The mode of operation to be used.',
        default=MODE_SINGLE,
    )

    @abstractmethod
    def configure_arguments(self, cli: CommandLineInterface):
        """
        Must be implemented by subclasses in order to configure the command line interface according to the mode of
        operation.

        :param cli: The command line interface to be configured
        """

    @abstractmethod
    def run_experiment(self, args: Namespace, experiment_builder_factory: Experiment.Builder.Factory):
        """
        Must be implemented by subclasses in order to run an experiment according to the command line arguments
        specified by the user.

        :param args:                        The command line arguments specified by the user
        :param experiment_builder_factory:  A factory function that allows to create instance of type
                                            `Experiment.Builder` that can be used for running experiments
        """
