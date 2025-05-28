"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing extensions that add functionality to the command line API provided by this software
package.
"""
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

from mlrl.testbed.experiments.experiment import Experiment


class Extension(ABC):
    """
    An abstract base class for all classes that add functionality to the command line API.
    """

    @abstractmethod
    def configure_arguments(self, argument_parser: ArgumentParser):
        """
        Must be implemented by subclasses in order to configure a given argument parser.

        :param argument_parser: The argument parser to be configured
        """

    @abstractmethod
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        Must be implemented by subclasses in order to configure an experiment according to the command line arguments
        specified by the user.

        :param args:                The command line arguments specified by the user
        :param experiment_builder:  A builder that allows to configure the experiment
        """
