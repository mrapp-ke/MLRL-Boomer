"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for implementing extensions that add functionality to the command line API provided by this software
package.
"""
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List

from mlrl.testbed.cli import Argument
from mlrl.testbed.experiments.experiment import Experiment


class Extension(ABC):
    """
    An abstract base class for all classes that add functionality to the command line API.
    """

    @abstractmethod
    def get_arguments(self) -> List[Argument]:
        """
        Must be implemented by subclasses in order to return the arguments that should be added to the command line API.

        :return: A list that contains the argument that should be added to the command line API
        """

    @abstractmethod
    def configure_experiment(self, args: Namespace, experiment_builder: Experiment.Builder):
        """
        Must be implemented by subclasses in order to configure an experiment according to the command line arguments
        specified by the user.

        :param args:                The command line arguments specified by the user
        :param experiment_builder:  A builder that allows to configure the experiment
        """
