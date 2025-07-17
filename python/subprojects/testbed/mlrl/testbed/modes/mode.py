"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides base classes for implementing different modes of operation.
"""
from abc import ABC, abstractmethod

from mlrl.util.cli import CommandLineInterface


class Mode(ABC):
    """
    An abstract base class for all modes of operation.
    """

    @abstractmethod
    def configure_arguments(self, cli: CommandLineInterface):
        """
        Must be implemented by subclasses in order to configure the command line interface according to the mode of
        operation.

        :param cli: The command line interface to be configured
        """
