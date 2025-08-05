"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)

Provides classes for setting up experiments.
"""
from abc import ABC, abstractmethod
from argparse import Namespace

from mlrl.testbed.command import Command
from mlrl.testbed.experiments.experiment import Experiment
from mlrl.testbed.experiments.input.dataset.splitters import DatasetSplitter
from mlrl.testbed.experiments.problem_domain import ProblemDomain


class Recipe(ABC):
    """
    An abstract base class for all classes that provide access to the ingredients that are needed by different
    operational modes for setting up experiments.
    """

    @abstractmethod
    def create_problem_domain(self, args: Namespace) -> ProblemDomain:
        """
        Creates and returns the `ProblemDomain` to be used in experiments.

        :param args:    The command line arguments specified by the user
        :return:        The `ProblemDomain` that has been created
        """

    @abstractmethod
    def create_dataset_splitter(self, args: Namespace) -> DatasetSplitter:
        """
        Creates and returns the `DatasetSplitter` to be used in experiments.

        :param args:    The command line arguments specified by the user
        :return:        The `DatasetSplitter` that has been created
        """

    @abstractmethod
    def create_experiment_builder(self, args: Namespace, command: Command) -> Experiment.Builder:
        """
        Creates and returns the `Experiment.Builder` to be used for configuring experiments.

        :param args:    The command line arguments specified by the user
        :param command: The command that has been used to start the experiment
        :return:        The `Experiment.Builder` that has been created
        """
