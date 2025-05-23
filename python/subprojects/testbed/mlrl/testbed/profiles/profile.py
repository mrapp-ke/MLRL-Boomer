"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing profiles that add functionality to the command line API provided by this software
package.
"""
from abc import ABC, abstractmethod
from argparse import ArgumentParser


class Profile(ABC):
    """
    An abstract base class for all classes that add functionality to the command line API.
    """

    @abstractmethod
    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        Must be implemented by subclasses in order to configure a given argument parser.

        :param argument_parser: The argument parser to be configured
        """
