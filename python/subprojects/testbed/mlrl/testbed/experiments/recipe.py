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
from mlrl.testbed.experiments.state import ExperimentMode


class Recipe(ABC):
    """
    An abstract base class for all classes that provide access to the ingredients that are needed by different
    operational modes for setting up experiments.
    """

    @abstractmethod
    def create_problem_domain(self, mode: ExperimentMode, args: Namespace) -> ProblemDomain:
        """
        Creates and returns the `ProblemDomain` to be used in experiments.

        :param mode:    The mode of operation
        :param args:    The command line arguments specified by the user
        :return:        The `ProblemDomain` that has been created
        """

    @abstractmethod
    def create_dataset_splitter(self, args: Namespace, load_dataset: bool = True) -> DatasetSplitter:
        """
        Creates and returns the `DatasetSplitter` to be used in experiments.

        :param args:            The command line arguments specified by the user
        :param load_dataset:    True, if the dataset should be loaded by the `DatasetSplitter`, False otherwise
        :return:                The `DatasetSplitter` that has been created
        """

    @abstractmethod
    def create_experiment_builder(self,
                                  experiment_mode: ExperimentMode,
                                  args: Namespace,
                                  command: Command,
                                  load_dataset: bool = True) -> Experiment.Builder:
        """
        Creates and returns the `Experiment.Builder` to be used for configuring experiments.

        :param experiment_mode: The mode of operation
        :param args:            The command line arguments specified by the user
        :param command:         The command that has been used to start the experiment
        :param load_dataset:    True, if the dataset should be loaded by the experiment, False otherwise
        :return:                The `Experiment.Builder` that has been created
        """
